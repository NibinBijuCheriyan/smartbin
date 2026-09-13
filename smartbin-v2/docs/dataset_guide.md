# SmartBin AI v2 — Dataset Engineering & Taxonomy Guide

## 1. Indian Waste Taxonomy (28 Categories)

The SmartBin AI v2 dataset is specifically tailored to municipal, commercial, and household waste streams commonly found in India. Items are grouped into four physical segregation compartments:

| Class ID | Class Name | Material Category | Physical Compartment | Recyclability |
|:---:|:---|:---|:---:|:---|
| **0** | `aluminium_can` | Aluminium Metal | **Recyclable** | Highly Recyclable |
| **1** | `banana_peel` | Organic Compostable | **Compost** | 100% Biodegradable |
| **2** | `biscuit_wrapper` | Multi-Layer Plastic (MLP) | **Recyclable** | Pyrolysis / Energy Recovery |
| **3** | `broken_plastic_toys`| Mixed Composite Plastic | **Landfill** | Non-recyclable |
| **4** | `cardboard` | Corrugated Fiberboard | **Recyclable** | High Value Pulp |
| **5** | `ceramic` | Inert Crockery | **Landfill** | Inert Landfill |
| **6** | `chips_packet` | Metallized BOPP Foil | **Recyclable** | Cement Kiln Refuse Fuel |
| **7** | `cigarette_butt` | Cellulose Acetate | **Landfill** | Hazardous Non-biodegradable |
| **8** | `coconut_shell` | Lignocellulose Husk | **Compost** | Biomass Fuel / Compost |
| **9** | `egg_shell` | Calcium Carbonate | **Compost** | Soil Conditioner |
| **10** | `fruit_waste` | Organic Compostable | **Compost** | Wet Waste Digestion |
| **11** | `glass_bottle` | Soda-lime Glass | **Recyclable** | Infinitely Recyclable |
| **12** | `glass_jar` | Soda-lime Glass | **Recyclable** | Infinitely Recyclable |
| **13** | `hdpe_bottle` | High-Density Polyethylene | **Recyclable** | High Value Plastic Flakes |
| **14** | `kurkure_packet` | Extruded Snack Foil | **Recyclable** | MLP Recycling |
| **15** | `leaves` | Yard Trimmings | **Compost** | Dry Organic Compost |
| **16** | `mask` | Polypropylene Non-woven | **Landfill** | Bio-waste Landfill |
| **17** | `milk_packet` | LDPE Co-extruded Pouch | **Recyclable** | Mechanical Recycling |
| **18** | `newspaper` | Newsprint | **Recyclable** | Paper Pulp |
| **19** | `paper_cup` | Poly-coated Chai Cup | **Recyclable** | Specialized Pulping |
| **20** | `pet_bottle` | Polyethylene Terephthalate | **Recyclable** | High Value rPET Yarn |
| **21** | `plastic_carry_bag` | Single-use Polybag | **Recyclable** | Low Value Film |
| **22** | `rice_waste` | Cooked Cereal Starch | **Compost** | Wet Kitchen Waste |
| **23** | `sanitary_waste` | Biowaste / Composite | **Landfill** | Incineration |
| **24** | `shampoo_bottle` | HDPE / PP Plastic | **Recyclable** | Rigid Plastic |
| **25** | `styrofoam` | Expanded Polystyrene | **Landfill** | Bulky Inert |
| **26** | `tea_bag` | Cellulose & Tea Powder | **Compost** | Wet Compost |
| **27** | `tetra_pak` | Aseptic 6-layer Paper/Alu | **Recyclable** | Paper Board / Composite |
| **28** | `tin_can` | Tin-plated Steel | **Recyclable** | Scrap Metal |
| **29** | `tissue` | Soiled Paper Napkin | **Landfill** | Contaminated Cellulosic |
| **30** | `vegetable_waste` | Raw Kitchen Scraps | **Compost** | High Moisture Wet Waste |
| **31** | `water_sachet` | LDPE Drinking Water Pouch | **Recyclable** | Flexible Plastic Film |

---

## 2. Ingestion & Format Normalization

The automated dataset builder merges data from:
1. **TrashNet** (Stanford)
2. **TACO** (In-the-wild litter detection)
3. **WasteNet & DeepWaste**
4. **Open Images V7** (Waste classes)
5. **Kaggle Garbage Classification**
6. **SmartBin Indian Municipal Custom Dataset**

Run the automated ingestion pipeline:

```bash
python -m smartbin_v2.dataset_builder.build_dataset --config smartbin-v2/configs/dataset_config.yaml
```

The pipeline automatically:
- Converts annotations to standard normalized YOLO format
- Eliminates visual duplicates via difference hashing (`dHash <= 4`)
- Balances class counts and creates stratified splits (70% Train, 15% Val, 15% Test)
- Generates bounding box spatial density heatmaps and class frequency charts.
