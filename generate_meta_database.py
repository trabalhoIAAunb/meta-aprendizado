from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pymfe.mfe import MFE
import csv
import math

np.__version__
pd.__version__

# Carrega o arquivo .arff e retorna um ndarray (Numpy)
def load_dataset(dataset_name):
    raw_data = arff.loadarff(dataset_name)
    dataframe = pd.DataFrame(raw_data[0])
    return dataframe.to_numpy()

# Separa atributos preditivos e atributos-alvo em um ndarray (Numpy)
def split_pred_target_feats(dataset):
    target_feat_pos = dataset.shape[1] - 1
    pred_data = dataset[:,:target_feat_pos]
    target_data = dataset[:,target_feat_pos]
    return pred_data, target_data

# Separa atributos preditivos de atributos-alvo e normaliza seus valores
def split_standardize_dataset(dataset):
    scaler = StandardScaler()
    pred_data, target_data = split_pred_target_feats(dataset)
    std_pred_data = scaler.fit_transform(pred_data)
    std_target_data = target_data.astype(int)
    return std_pred_data, std_target_data

# Calcula a média das métricas de performance para o k conjuntos de treino/teste
def mean_scores(scores):
    return np.mean(scores['test_accuracy']), np.mean(scores['test_f1'])

# Seleciona o melhor entre 2 algoritmos de classificação a partir da média das métricas de performance
def get_best_classifier(val1, val2):
    max_value = max(val1, val2)
    if max_value == val1: return 1
    else: return 2

# Obtém os meta-atributos das bases de dados (preditivos e alvo), utilizando KNN e árvore de decisão
def get_meta_feats(dataset_name):
    raw_dataset = load_dataset('./bases_de_dados/'+dataset_name)
    x, y = split_standardize_dataset(raw_dataset)

    meta_feats_extractor = MFE()
    meta_feats_extractor.fit(x, y)
    meta_feats = meta_feats_extractor.extract()
    x_meta_feats = meta_feats[1]

    classifier_1 = DecisionTreeClassifier()
    scores_1 = cross_validate(estimator=classifier_1, X=x, y=y, scoring=('accuracy','f1'), cv=10)
    accuracy_1, f1_1 = mean_scores(scores_1)

    classifier_2 = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
    scores_2 = cross_validate(estimator=classifier_2, X=x, y=y, scoring=('accuracy','f1'), cv=10)
    accuracy_2, f1_2 = mean_scores(scores_2)

    best_accuracy_classifier = get_best_classifier(accuracy_1, accuracy_2)
    best_f1_classifier = get_best_classifier(f1_1,f1_2)
    
    meta_feats_with_accuracy = x_meta_feats.copy()
    meta_feats_with_accuracy.append(best_accuracy_classifier)
    meta_feats_with_f1 = x_meta_feats.copy()
    meta_feats_with_f1.append(best_f1_classifier)

    return meta_feats_with_accuracy, meta_feats_with_f1

# Gera arquivo .npy com os meta-atributos
def generate_npy_file(meta_database):
    np.save(f"./meta_database",meta_database)

# Gera arquivo .csv com os meta-atributos
def generate_csv_file(meta_database):
    all_meta_feat_types = ['attr_conc.mean', 'attr_conc.sd', 'attr_ent.mean', 'attr_ent.sd', 'attr_to_inst', 'best_node.mean', 'best_node.sd', 'can_cor.mean', 'can_cor.sd', 'cat_to_num', 'class_conc.mean', 'class_conc.sd', 'class_ent', 'cor.mean', 'cor.sd', 'cov.mean', 'cov.sd', 'eigenvalues.mean', 'eigenvalues.sd', 'elite_nn.mean', 'elite_nn.sd', 'eq_num_attr', 'freq_class.mean', 'freq_class.sd', 'g_mean.mean', 'g_mean.sd', 'gravity', 'h_mean.mean', 'h_mean.sd', 'inst_to_attr', 'iq_range.mean', 'iq_range.sd', 'joint_ent.mean', 'joint_ent.sd', 'kurtosis.mean', 'kurtosis.sd', 'leaves', 'leaves_branch.mean', 'leaves_branch.sd', 'leaves_corrob.mean', 'leaves_corrob.sd', 'leaves_homo.mean', 'leaves_homo.sd', 'leaves_per_class.mean', 'leaves_per_class.sd', 'lh_trace', 'linear_discr.mean', 'linear_discr.sd', 'mad.mean', 'mad.sd', 'max.mean', 'max.sd', 'mean.mean', 'mean.sd', 'median.mean', 'median.sd', 'min.mean', 'min.sd', 'mut_inf.mean', 'mut_inf.sd', 'naive_bayes.mean', 'naive_bayes.sd', 'nodes', 'nodes_per_attr', 'nodes_per_inst', 'nodes_per_level.mean', 'nodes_per_level.sd', 'nodes_repeated.mean', 'nodes_repeated.sd', 'nr_attr', 'nr_bin', 'nr_cat', 'nr_class', 'nr_cor_attr', 'nr_disc', 'nr_inst', 'nr_norm', 'nr_num', 'nr_outliers', 'ns_ratio', 'num_to_cat', 'one_nn.mean', 'one_nn.sd', 'p_trace', 'random_node.mean', 'random_node.sd', 'range.mean', 'range.sd', 'roy_root', 'sd.mean', 'sd.sd', 'sd_ratio', 'skewness.mean', 'skewness.sd', 'sparsity.mean', 'sparsity.sd', 't_mean.mean', 't_mean.sd', 'tree_depth.mean', 'tree_depth.sd', 'tree_imbalance.mean', 'tree_imbalance.sd', 'tree_shape.mean', 'tree_shape.sd', 'var.mean', 'var.sd', 'var_importance.mean', 'var_importance.sd', 'w_lambda', 'worst_node.mean', 'worst_node.sd','pred_classifier']
    with open('./meta_database.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(all_meta_feat_types)
        write.writerows(meta_database)

databases_names = ['1004_synthetic_control.arff', '1005_glass.arff', '1011_ecoli.arff', '1015_confidence.arff', '1020_mfeat-karhunen.arff', '1021_page-blocks.arff', '1048_jEdit_4.2_4.3.arff', '1049_pc4.arff', '1050_pc3.arff', '1054_mc2.arff', '1056_mc1.arff', '1061_ar4.arff', '1064_ar6.arff', '1065_kc3.arff', '1066_kc1-binary.arff', '1068_pc1.arff', '1069_pc2.arff', '1071_mw1.arff', '1073_jEdit_4.0_4.2.arff', '1075_datatrieve.arff', '1460_banana.arff', '1462_banknote-authentication.arff', '1464_blood-transfusion-service-center.arff', '1467_climate-model-simulation-crashes.arff', '1473_fertility.arff', '1479_hill-valley.arff', '1487_ozone-level-8hr.arff', '1488_parkinsons.arff', '1489_phoneme.arff', '1490_planning-relax.arff', '1494_qsar-biodeg.arff', '1496_ringnorm.arff', '1504_steel-plates-fault.arff', '1507_twonorm.arff', '1510_wdbc.arff', '1524_vertebra-column.arff', '1547_autoUniv-au1-1000.arff', '1566_hill-valley.arff', '1600_SPECTF.arff', '298_coil2000.arff', '337_SPECTF.arff', '37_diabetes.arff', '40_sonar.arff', '40665_clean1.arff', '40666_clean2.arff', '40704_Titanic.arff', '40705_tokyo1.arff', '40900_Satellite.arff', '40910_Speech.arff', '40983_wilt.arff', '40994_climate-model-simulation-crashes.arff', '44_spambase.arff', '464_prnn_synth.arff', '467_analcatdata_japansolvent.arff', '472_lupus.arff', '53_heart-statlog.arff', '59_ionosphere.arff', '713_vineyard.arff', '715_fri_c3_1000_25.arff', '716_fri_c3_100_50.arff', '717_rmftsa_ladata.arff', '718_fri_c4_1000_100.arff', '721_pwLinear.arff', '723_fri_c4_1000_25.arff', '726_fri_c2_100_5.arff', '728_analcatdata_supreme.arff', '729_visualizing_slope.arff', '730_fri_c1_250_5.arff', '731_baskball.arff', '732_fri_c0_250_50.arff', '733_machine_cpu.arff', '735_cpu_small.arff', '736_visualizing_environmental.arff', '737_space_ga.arff', '740_fri_c3_1000_10.arff', '742_fri_c4_500_100.arff', '743_fri_c1_1000_5.arff', '744_fri_c3_250_5.arff', '746_fri_c1_250_25.arff', '749_fri_c3_500_5.arff', '750_pm10.arff', '753_wisconsin.arff', '754_fri_c0_100_5.arff', '758_analcatdata_election2000.arff', '759_analcatdata_olympic2000.arff', '761_cpu_act.arff', '762_fri_c2_100_10.arff', '763_fri_c0_250_10.arff', '766_fri_c1_500_50.arff', '768_fri_c3_100_25.arff', '769_fri_c1_250_50.arff', '770_strikes.arff', '772_quake.arff', '773_fri_c0_250_25.arff', '774_disclosure_x_bias.arff', '775_fri_c2_100_25.arff', '777_sleuth_ex1714.arff', '778_bodyfat.arff', '779_fri_c1_500_25.arff', '780_rabe_265.arff', '782_rabe_266.arff', '783_fri_c3_100_10.arff', '785_wind_correlations.arff', '787_witmer_census_1980.arff', '788_triazines.arff', '789_fri_c1_100_10.arff', '791_diabetes_numeric.arff', '792_fri_c2_500_5.arff', '793_fri_c3_250_10.arff', '794_fri_c2_250_25.arff', '795_disclosure_x_tampered.arff', '797_fri_c4_1000_50.arff', '799_fri_c0_1000_5.arff', '800_pyrim.arff', '801_chscase_funds.arff', '803_delta_ailerons.arff', '805_fri_c4_500_50.arff', '806_fri_c3_1000_50.arff', '807_kin8nm.arff', '812_fri_c1_100_25.arff', '813_fri_c3_1000_5.arff', '814_chscase_vine2.arff', '815_chscase_vine1.arff', '817_diggle_table_a1.arff', '819_delta_elevators.arff', '820_chatfield_4.arff', '824_fri_c1_500_10.arff', '827_disclosure_x_noise.arff', '828_fri_c4_100_100.arff', '829_fri_c1_100_5.arff', '830_fri_c2_250_10.arff', '834_fri_c4_250_100.arff', '837_fri_c1_1000_50.arff', '838_fri_c4_500_25.arff', '841_stock.arff', '845_fri_c0_1000_10.arff', '847_wind.arff', '849_fri_c0_1000_25.arff', '850_fri_c0_100_50.arff', '851_tecator.arff', '855_fri_c4_500_10.arff', '857_bolts.arff', '859_analcatdata_gviolence.arff', '860_vinnie.arff', '863_fri_c4_250_10.arff', '866_fri_c2_1000_50.arff', '868_fri_c4_100_25.arff', '869_fri_c2_500_10.arff', '870_fri_c1_500_5.arff', '873_fri_c3_250_50.arff', '874_rabe_131.arff', '879_fri_c2_500_25.arff', '882_pollution.arff', '884_fri_c0_500_5.arff', '885_transplant.arff', '886_no2.arff', '888_fri_c0_500_50.arff', '889_fri_c0_100_25.arff', '892_sleuth_case1201.arff', '893_visualizing_hamster.arff', '895_chscase_geyser1.arff', '896_fri_c3_500_25.arff', '900_chscase_census6.arff', '903_fri_c2_1000_25.arff', '905_chscase_adopt.arff', '906_chscase_census5.arff', '907_chscase_census4.arff', '908_chscase_census3.arff', '909_chscase_census2.arff', '910_fri_c1_1000_10.arff', '911_fri_c2_250_5.arff', '912_fri_c2_1000_5.arff', '913_fri_c2_1000_10.arff', '917_fri_c1_1000_25.arff', '918_fri_c4_250_50.arff', '920_fri_c2_500_50.arff', '922_fri_c2_100_50.arff', '925_visualizing_galaxy.arff', '926_fri_c0_500_25.arff', '927_hutsof99_child_witness.arff', '931_disclosure_z.arff', '933_fri_c4_250_25.arff', '936_fri_c3_500_10.arff', '937_fri_c3_500_50.arff', '943_fri_c0_500_10.arff', '946_visualizing_ethanol.arff', '958_segment.arff', '962_mfeat-morphological.arff', '969_iris.arff', '970_analcatdata_authorship.arff', '971_mfeat-fourier.arff', '973_wine.arff', '974_hayes-roth.arff', '978_mfeat-factors.arff', '979_waveform-5000.arff', '980_optdigits.arff', '994_vehicle.arff', '995_mfeat-zernike.arff', '997_balance-scale.arff']
meta_database_with_accuracy = []
meta_database_with_f1 = []
for i in databases_names:
    a, b = get_meta_feats(i)
    meta_database_with_accuracy.append(a)
    meta_database_with_f1.append(b)

generate_npy_file(meta_database_with_accuracy)
generate_csv_file(meta_database_with_accuracy)