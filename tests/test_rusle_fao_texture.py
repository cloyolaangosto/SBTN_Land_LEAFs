"""Tests for the FAO texture class (Jones, Spoor & Thomasson 2003, Table 1) and the structure class s in sbtn_leaf.rusle."""

import numpy as np

from sbtn_leaf import rusle


def test_fao_texture_class_boundaries():
    #               coarse  coarse-edge medium medium(>65 sand, 18 clay) medium fine fine  fine-edge very fine
    sand = np.array([80,    66,         40,    70,                       10,         30,   30,       10], dtype=np.float32)
    clay = np.array([10,    17,         20,    18,                       30,         40,   60,       65], dtype=np.float32)
    assert rusle._determine_fao_texture_class(sand, clay).tolist() == [1, 1, 2, 2, 3, 4, 4, 5]


def test_fao_texture_class_nodata():
    sand = np.array([np.nan, 50], dtype=np.float32)
    clay = np.array([20, 20], dtype=np.float32)
    assert rusle._determine_fao_texture_class(sand, clay).tolist() == [0, 2]
    assert rusle._determine_fao_texture_class(sand, clay, valid=np.array([True, False])).tolist() == [0, 0]


def test_structure_class_from_usda_function():
    # sand, silt, clay -> FAO coarse, medium, medium fine, fine, very fine, nodata
    sand = np.array([85, 40, 10, 30, 10, 0], dtype=np.float32)
    silt = np.array([10, 40, 70, 30, 25, 0], dtype=np.float32)
    clay = np.array([5, 20, 20, 40, 65, 0], dtype=np.float32)
    texture, s, p = rusle._determine_usda_class_number_permeability(sand, silt, clay)
    assert s.tolist() == [1, 2, 2, 4, 4, 0]
    assert s.dtype == np.uint8 and p.dtype == np.uint8
    assert texture[-1] == 0 and p[-1] == 0
