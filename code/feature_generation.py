import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.element import ElementFraction
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.composition.alloy import Miedema
from matminer.featurizers.composition.alloy import YangSolidSolution
from matminer.featurizers.composition.packing import AtomicPackingEfficiency
from matminer.featurizers.composition.alloy import WenAlloys
from matminer.featurizers.composition.orbital import ValenceOrbital
from matminer.featurizers.composition.composite import Meredig
from matminer.featurizers.composition.element import TMetalFraction
from matminer.featurizers.composition.element import Stoichiometry
from matminer.featurizers.composition.element import BandCenter


def feature_generation(df, column_name="Formula [at.%]"):
    df_fg = pd.DataFrame({})

    stc = StrToComposition()
    data_with_composition = stc.featurize_dataframe(df, column_name)

    com_ele_mag = ElementProperty.from_preset("magpie_less")
    df_com_ele_mag = com_ele_mag.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_ele_mag = df_com_ele_mag.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_ele_mag.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_mie = Miedema()
    df_com_mie = com_mie.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_mie = df_com_mie.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_mie.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_yan = YangSolidSolution()
    df_com_yan = com_yan.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_yan = df_com_yan.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_yan.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_wen = WenAlloys()
    df_com_wen = com_wen.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_wen = df_com_wen.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_wen.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_ele_dem = ElementProperty.from_preset("deml_less")
    df_com_ele_dem = com_ele_dem.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_ele_dem = df_com_ele_dem.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_ele_dem.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_ele_mat = ElementProperty.from_preset("matminer_less")
    df_com_ele_mat = com_ele_mat.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_ele_mat = df_com_ele_mat.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_ele_mat.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    orb_val = ValenceOrbital()
    df_orb_val = orb_val.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_orb_val = df_orb_val.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_orb_val.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    com_mer = Meredig()
    df_com_mer = com_mer.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_com_mer = df_com_mer.iloc[:, 105:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_com_mer.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    ele_tme = TMetalFraction()
    df_ele_tme = ele_tme.featurize_dataframe(
        data_with_composition, "composition", ignore_errors=False
    )
    df_ele_tme = df_ele_tme.iloc[:, 2:]
    df_fg = pd.concat(
        [df_fg.reset_index(drop=True), df_ele_tme.reset_index(drop=True)], axis=1
    )
    print(df_fg.shape)

    keep_columns = [
        "MagpieData mean AtomicWeight",
        "MagpieData std_dev AtomicWeight",
        "MagpieData mean CovalentRadius",
        "MagpieData std_dev CovalentRadius",
        "MagpieData mean Electronegativity",
        "MagpieData std_dev Electronegativity",
        "MagpieData mean NValence",
        "MagpieData std_dev NValence",
        "MagpieData mean NUnfilled",
        "MagpieData std_dev NUnfilled",
        "Yang omega",
        "Yang delta",
        "Radii local mismatch",
        "Radii gamma",
        "Electronegativity delta",
        "Electronegativity local mismatch",
        "VEC mean",
        "Mean cohesive energy",
        "Interant electrons",
        "Shear modulus mean",
        "Shear modulus delta",
        "Shear modulus local mismatch",
        "Shear modulus strength model",
        "DemlData mean atom_radius",
        "DemlData std_dev atom_radius",
        "DemlData mean molar_vol",
        "DemlData std_dev molar_vol",
        "DemlData mean heat_fusion",
        "DemlData std_dev heat_fusion",
        "DemlData mean heat_cap",
        "DemlData std_dev heat_cap",
        "DemlData mean first_ioniz",
        "DemlData std_dev first_ioniz",
        "PymatgenData mean electrical_resistivity",
        "PymatgenData std_dev electrical_resistivity",
        "PymatgenData mean velocity_of_sound",
        "PymatgenData std_dev velocity_of_sound",
        "PymatgenData mean thermal_conductivity",
        "PymatgenData std_dev thermal_conductivity",
        "PymatgenData mean bulk_modulus",
        "PymatgenData std_dev bulk_modulus",
        "PymatgenData mean coefficient_of_linear_thermal_expansion",
        "PymatgenData std_dev coefficient_of_linear_thermal_expansion",
        "transition metal fraction"
    ]
    
    df_fg = df_fg[keep_columns]
    print(df_fg.shape)

    column_names = df_fg.columns.tolist()
    df_fg_col = pd.DataFrame(column_names, columns=["Column Names"])

    return df_fg.reset_index(drop=True), df_fg_col.reset_index(drop=True)


def feature_generation_calphad(df, column_name="Formula [at.%]"):
    df_fg, df_fg_col = feature_generation(df, column_name)

    column_list = [
        "MagpieData std_dev Electronegativity",
        "MagpieData std_dev NValence",
        "MagpieData std_dev NUnfilled",
        "Radii gamma",
        "Electronegativity delta",
        "DemlData std_dev atom_radius",
        "DemlData std_dev molar_vol",
    ]
    df_fg = df_fg[column_list]

    print(df_fg.shape)

    column_names = df_fg.columns.tolist()
    df_fg_col = pd.DataFrame(column_names, columns=["Column Names"])

    return df_fg, df_fg_col