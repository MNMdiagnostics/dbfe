#!/usr/bin/env python3
# Example:
#	python3 get_variant_lengths.py /mnt/computations/breast_cancer_results breast

import pandas as pd
import numpy as np
import glob
import gzip
import os
import sys

chrom_default_lengths = {
    "1": 248956422,
    "2": 242193529,
    "3": 198295559,
    "4": 190214555,
    "5": 181538259,
    "6": 170805979,
    "7": 159345973,
    "8": 145138636,
    "9": 138394717,
    "10": 133797422,
    "11": 135086622,
    "12": 133275309,
    "13": 114364328,
    "14": 107043718,
    "15": 101991189,
    "16": 90338345,
    "17": 83257441,
    "18": 80373285,
    "19": 58617616,
    "20": 64444167,
    "21": 46709983,
    "22": 50818468,
    "X": 156040895,
    "Y": 57227415,
    "MT": 16569,
}


def search_and_combine(input_folder, file_suffix, parse_func, output_file_name):
    print("Searching for", file_suffix, "files")
    files = glob.glob(input_folder + '/*' + file_suffix, recursive=False)
    files_num = len(files)
    print("Found", str(files_num), "files")

    combined_df = pd.DataFrame()
    output_df_file = output_file_name + '.csv.gz'
    i = 0

    for file in files:
        if i % 100 == 0:
            print(str(i), "/", str(files_num))

        try:
            df = parse_func(file)
            combined_df = combined_df.append(df)
        except Exception as e:
            print("Could not parse ", file, e)
        except:
            print("Unknown problem with", file)

        i += 1

    print("Saving combined data to file:", output_df_file)
    combined_df.to_csv(output_df_file, index=False, compression='gzip')

    return combined_df.shape[0]


def parse_ascat(file):
    ASCAT_COLUMNS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "NORMAL", "TUMOUR"]

    df = pd.read_csv(file, sep="\t", comment="#", header=None, names=ASCAT_COLUMNS)
    df.loc[:, "END"] = df.INFO.str.split('END=').str[1].astype(int)
    df.loc[:, "LEN"] = df.END - df.POS + 1
    assert len(df.FORMAT.unique()) == 1
    assert df.FORMAT[0] == "GT:TCN:MCN"
    df.loc[:, "TCN_NORMAL"] = df.NORMAL.str.split(":").str[1].astype(int)
    df.loc[:, "TCN_TUMOUR"] = df.TUMOUR.str.split(":").str[1].astype(int)
    df.loc[:, "TCN_DIFF"] = df.loc[:, "TCN_TUMOUR"] - df.loc[:, "TCN_NORMAL"]
    df.loc[:, "SVCLASS"] = "DIP"
    df.loc[df.TCN_DIFF < 0, "SVCLASS"] = "DEL"
    df.loc[df.TCN_DIFF > 0, "SVCLASS"] = "DUP"
    df.loc[:, "SAMPLEID"] = os.path.basename(file).split("_vs_")[0]
    df = df.loc[:, ["CHROM", "LEN", "SVCLASS", "SAMPLEID"]]

    return df


def parse_brass(file):
    BRASS_COLUMNS = ["chr1", "start1", "end1", "chr2", "start2", "end2", "id/name", "brass_score",
                     "strand1_1", "strand2_1", "sample", "svclass", "bkdist", "assembly_score",
                     "readpair names", "readpair count", "bal_trans", "inv", "occL", "occH",
                     "copynumber_flag", "range_blat", "Brass Notation", "non-template", "micro-homology",
                     "assembled readnames", "assembled read count", "gene1", "gene_id1", "transcript_id1",
                     "strand1_2", "end_phase1", "region1", "region_number1", "total_region_count1",
                     "first/last1", "gene2", "gene_id2", "transcript_id2", "strand2_2", "phase2",
                     "region2", "region_number2", "total_region_count2", "first/last2", "fusion_flag"]

    df = pd.read_csv(file, sep="\t", comment="#", header=None, names=BRASS_COLUMNS)
    df.loc[:, "CHROM"] = df.chr1

    #     https://dermasugita.github.io/ViolaDocs/docs/html/userguide/sv_position_specification.html
    df.loc[:, "LEN"] = df.bkdist.astype(int) + 1
    df.loc[:, "end1_backwards"] = 0
    df.loc[:, "end2_backwards"] = 0
    for chrom1 in df.chr1.unique():
        df.loc[df.chr1 == chrom1, "end1_backwards"] = np.abs(chrom_default_lengths[str(chrom1)] - df.loc[df.chr1 == chrom1, "end1"])
    for chrom2 in df.chr2.unique():
        df.loc[df.chr2 == chrom2, "end2_backwards"] = np.abs(chrom_default_lengths[str(chrom2)] - df.loc[df.chr2 == chrom2, "end2"])
    df.loc[df.LEN == 0, "LEN"] = df.loc[df.LEN == 0, ['end1', 'end2', "end1_backwards", "end2_backwards"]].min(axis=1)

    df.loc[:, "SVCLASS"] = df.svclass
    df.loc[:, "SAMPLEID"] = os.path.basename(file).split("_vs_")[0]
    df = df.loc[:, ["CHROM", "LEN", "SVCLASS", "SAMPLEID"]]

    return df


def parse_manta(file):
    df = manta_vcf_to_bedpe(file)

    if df is None:
        raise Exception("Empty VCF file")

    df.loc[:, "CHROM"] = df.chrom1
    df.loc[df.SVTYPE != "TRA", "LEN"] = df.loc[df.SVTYPE != "TRA", "start2"] - df.loc[df.SVTYPE != "TRA", "stop1"]

    # translocation lengths
    chrom_lengths = try_get_chrom_lengths(file)
    df.loc[df.SVTYPE == "TRA", "LEN"] = np.nan
    df.loc[:, "stop1_backwards"] = 0
    df.loc[:, "stop2_backwards"] = 0
    for c1 in df.chrom1.unique():
        df.loc[(df.SVTYPE == "TRA") & (df.chrom1 == c1), "stop1_backwards"] = chrom_lengths[str(c1)] - df.loc[(df.SVTYPE == "TRA") & (df.chrom1 == c1), "stop1"]
    for c2 in df.chrom2.unique():
        df.loc[(df.SVTYPE == "TRA") & (df.chrom2 == c2), "stop2_backwards"] = chrom_lengths[str(c2)] - df.loc[(df.SVTYPE == "TRA") & (df.chrom2 == c2), "stop2"]
    df.loc[df.LEN.isna(), "LEN"] = df.loc[df.LEN.isna(), ['stop1', 'stop2', "stop1_backwards", "stop2_backwards"]].min(axis=1)

    df = df.loc[df.LEN != 0, :]
    df.LEN = np.abs(df.LEN)

    df.loc[:, "SVCLASS"] = df.SVTYPE
    df.loc[:, "SAMPLEID"] = os.path.basename(file).split("_vs_")[0]
    df = df.loc[:, ["CHROM", "LEN", "SVCLASS", "SAMPLEID"]]

    return df


def parse_strelka(file):
    STRELKA_COLUMNS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "NORMAL", "TUMOR"]
    STRELKA_DTYPES = {'CHROM': np.dtype('O'), 'POS': np.dtype('int64'), 'ID': np.dtype('O'), 'REF': np.dtype('O'), 'ALT': np.dtype('O'), 'QUAL': np.dtype('O'), 'FILTER': np.dtype('O'), 'INFO': np.dtype('O'), 'FORMAT': np.dtype('O'), 'NORMAL': np.dtype('O'), 'TUMOR': np.dtype('O')}
    df = pd.read_csv(file, sep="\t", comment="#", header=None, names=STRELKA_COLUMNS, dtype=STRELKA_DTYPES)
    df.loc[:, "DIFF"] = df.ALT.str.len() - df.REF.str.len()
    df.loc[:, "LEN"] = df.loc[:, "DIFF"].abs()
    df.loc[:, "SVCLASS"] = "INS"
    df.loc[df.loc[:, "DIFF"] < 0, "SVCLASS"] = "DEL"
    df.loc[:, "SAMPLEID"] = os.path.basename(file).split("_vs_")[0]
    df = df.loc[:, ["CHROM", "LEN", "SVCLASS", "SAMPLEID"]]

    return df


def try_get_chrom_lengths(file):
    chrom_lengths = {}

    for chrom, chrom_default_length in chrom_default_lengths.items():
        with gzip.open(file, "rt") as file_text:
            for line in file_text:
                if line.startswith("##contig=<ID=" + chrom):
                    chrom_lengths[chrom] = int(line.split("length=")[1].strip()[:-1])
                    break
        if not chrom in chrom_lengths:
            chrom_lengths[chrom] = chrom_default_length

    return chrom_lengths


def manta_vcf_to_bedpe(file):
    MANTA_COLUMNS = ['chrom1', 'pos1', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format', 'normal', 'tumor']
    vcf = pd.read_csv(file, sep="\t", comment='#', names=MANTA_COLUMNS)

    ##############3
    # This code below is Alex's vcf->bedpe script
    #########
    # filters, additional better to add here, make a dataframe for calculation of Confidence intervals(CIs) for each event
    CIs = vcf.loc[vcf["filter"] == "PASS"]
    # Divede into breakends(DUP,INS,DEL) and breakpoints(BND)
    CI_dup_del = CIs.loc[
        (CIs['info'].str.contains("=DUP", case=True)) | (CIs['info'].str.contains("=DEL", case=True)) | (
            CIs['info'].str.contains("=INS", case=True))]
    CI_other = CIs.loc[(CIs['info'].str.contains("=BND", case=True))]
    # print(CIs.shape)
    # print(CI_dup_del.shape)
    # print(CI_other.shape)
    if (CIs.shape[0] == 0):
        return None
    if (CI_dup_del.shape[0] != 0):
        # Breakpionts
        def extract_info(info):
            if "CIPOS" in info:
                CI = info.split("CIPOS=")[1].split(";")[0]
                CI_1 = int(CI.split(",")[0])
                CI_2 = int(CI.split(",")[1])
            else:
                CI_1 = int(0)
                CI_2 = int(0)
            if "CIEND" in info:
                CIE = info.split("CIEND=")[1].split(";")[0]
                CIE_1 = int(CIE.split(",")[1])
                CIE_2 = int(CIE.split(",")[1])
            else:
                CIE_1 = int(0)
                CIE_2 = int(0)
            SVTYPE = info.split("SVTYPE=")[1].split(";")[0]
            END = int(info.split("END=")[1].split(";")[0])
            try:
                HOMSEQ = info.split("HOMSEQ=")[1].split(";")[0]
            except:
                HOMSEQ = None
            return (CI_1, CI_2, CIE_1, CIE_2, SVTYPE, END, HOMSEQ)

        CI_dup_del = CI_dup_del[["chrom1", "pos1", "ref", "alt", "id", "info"]].copy()
        CI_dup_del[["CI_1", "CI_2", "CIE_1", "CIE_2", "SVTYPE", "END", "HOMSEQ1"]] = [extract_info(x) for x in
                                                                                      CI_dup_del['info']]
        CI_dup_del["start1"] = CI_dup_del.apply(lambda row: int(row['pos1'] - abs(row['CI_1']) - 1), axis=1)
        CI_dup_del["stop1"] = CI_dup_del.apply(lambda row: int(row['pos1'] + row['CI_2']), axis=1)
        CI_dup_del["start2"] = CI_dup_del.apply(lambda row: int(row['END'] - abs(row['CIE_1']) - 1), axis=1)
        CI_dup_del["stop2"] = CI_dup_del.apply(lambda row: int(row['END'] + row['CIE_2']), axis=1)
        CI_dup_del["strand1"] = CI_dup_del.apply(
            lambda row: "+" if ((row['SVTYPE'] == "DEL") | (row['SVTYPE'] == "INS")) else "-", axis=1)
        CI_dup_del["strand2"] = CI_dup_del.apply(
            lambda row: "-" if ((row['SVTYPE'] == "DEL") | (row['SVTYPE'] == "INS")) else "+", axis=1)
        CI_dup_del["SAMPLE"] = os.path.basename(file).split('_vs_')[0]
        CI_dup_del["chrom2"] = CI_dup_del["chrom1"]
        CI_dup_del["HOMSEQ2"] = None
        CI_dup_del_final = CI_dup_del[
            ["chrom1", "start1", "stop1", "chrom2", "start2", "stop2", "strand1", "strand2", "SAMPLE", "SVTYPE",
             "HOMSEQ1", "HOMSEQ2"]]
    # print(CI_dup_del_final)
    else:
        pass
    # Breakends
    ############### IVERSIONS, TRANSLOCATIONs
    # print(CI_other)
    if (CI_other.shape[0] != 0):
        CI_other = CI_other[["chrom1", "pos1", "ref", "alt", "id", "info"]].copy()
        mate_set = set()

        def extract_info(info):
            if "CIPOS" in info:
                CI = info.split("CIPOS=")[1].split(";")[0]
                CI_1 = int(CI.split(",")[0])
                CI_2 = int(CI.split(",")[1])
            else:
                CI_1 = int(0)
                CI_2 = int(0)
            try:
                HOMSEQ = info.split("HOMSEQ=")[1].split(";")[0]
            except:
                HOMSEQ = None
            MID = info.split("MATEID=")[1].split(";")[0]
            return (CI_1, CI_2, MID, HOMSEQ)

        CI_other[["CI_1", "CI_2", "MATEID", "HOMSEQ"]] = [extract_info(x) for x in CI_other['info']]
        CI_other["start"] = CI_other.apply(lambda row: int(row['pos1'] - abs(row['CI_1']) - 1), axis=1)
        CI_other["stop"] = CI_other.apply(lambda row: int(row['pos1'] + row['CI_2']), axis=1)
        #### Coupling of mates
        # Oreder do not matter. To flip these calls is fairly straightforward; the ends of the intervals (5’ or 3’) belong to their respective breakpoint positions, so flipping the breakpoints just means flipping the positions, and the first and second number in the CT record. So 3to3 and 5to5 remain the same, while 3to5 becomes 5to3 and vice versa.
        ids_set = set()

        def pairs_divide(row):
            if row[0] not in ids_set:
                ids_set.add(row[1])
                return (1)
            else:
                return (0)

        CI_other["set"] = [pairs_divide(row) for row in zip(CI_other["id"], CI_other["MATEID"])]

        def alt_orientation(alt):
            return ('-' if (alt[0] == '[' or alt[0] == ']') else '+')

        CI_other["strand"] = [alt_orientation(x) for x in CI_other['alt']]
        CI_other = CI_other[["chrom1", "start", "stop", "strand", "id", "MATEID", "HOMSEQ", "set"]]
        CI_other.rename({'chrom1': 'chrom'}, axis=1, inplace=True)
        CI_other_1 = CI_other.loc[CI_other["set"] == 1].copy()
        CI_other_0 = CI_other.loc[CI_other["set"] == 0].copy()
        CI_other_0.drop('id', axis=1, inplace=True)
        CI_other_1.drop('MATEID', axis=1, inplace=True)
        CI_other = pd.merge(CI_other_1, CI_other_0, how='left', left_on=["id"], right_on=["MATEID"],
                            suffixes=('1', '2'))
        CI_other.drop(['id', "MATEID", "set1", "set2"], axis=1, inplace=True)

        def sv_type(row):
            if row[0] != row[1]:
                svtype = 'TRA'
            elif row[2] == row[3]:
                svtype = 'INV'
            return (svtype)

        CI_other["SAMPLE"] = os.path.basename(file).split('_vs_')[0]
        CI_other["SVTYPE"] = [sv_type(row) for row in
                              zip(CI_other["chrom1"], CI_other["chrom2"], CI_other["strand1"], CI_other["strand2"])]
        CI_other_final = CI_other[
            ["chrom1", "start1", "stop1", "chrom2", "start2", "stop2", "strand1", "strand2", "SAMPLE", "SVTYPE",
             "HOMSEQ1", "HOMSEQ2"]]
    # print(CI_other_final)
    else:
        pass
    if (CI_dup_del.shape[0] != 0) & (CI_other.shape[0] != 0):
        CI_final = CI_other_final.append(CI_dup_del_final)
    elif (CI_other.shape[0] == 0):
        CI_final = CI_dup_del_final
    elif (CI_dup_del.shape[0] == 0):
        CI_final = CI_other_final
    else:
        return None
    CI_final.sort_values(by=['chrom1', 'stop1'], inplace=True)
    chr_list = [str(int) for int in list(range(1, 23))]
    chr_list.extend(["X", "Y"])
    CI_final['chrom1'] = CI_final['chrom1'].astype(str)
    CI_final['chrom2'] = CI_final['chrom2'].astype(str)
    CI_final = CI_final[(CI_final["chrom1"].isin(chr_list)) & (CI_final["chrom2"].isin(chr_list))]

    return CI_final


if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_file = sys.argv[2]

    search_and_combine(input_folder, "ascat.vcf.gz", parse_ascat, output_file + "_cnv")
    search_and_combine(input_folder, "manta.somatic.vcf.gz", parse_manta, output_file + "_sv")
    search_and_combine(input_folder, "strelka.somatic.indels.vcf.gz", parse_strelka, output_file + "_indel")