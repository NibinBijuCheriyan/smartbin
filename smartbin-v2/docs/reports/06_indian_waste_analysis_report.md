# Research Report 06: Domain Analysis on Indian Municipal & Consumer Waste

## Executive Summary
Unlike Western benchmarks that predominantly feature rigid PET bottles and clean Amazon cartons, Indian municipal and consumer waste exhibits high moisture content, extensive multi-layer packaging (MLP), ubiquitous milk pouches, disposable chai cups, and organic food scraps. This report provides a domain analysis addressing these unique challenges.

---

## 1. Domain Specific Challenges in Indian Waste

### A. High Moisture Organic Kitchen Scraps
- **Characteristics**: Wet cooked rice, vegetable peelings, and coconut husks possess high moisture content (60–75%), causing specular highlights under camera illumination.
- **Solution**: Integrating CLAHE and random gamma transformations prevents white-out saturation over wet fruit and vegetable surfaces.

### B. Multi-Layer Packaging (MLP)
- **Characteristics**: Chips packets (Lay's, Bingo) and extruded snacks (Kurkure) use metallized bi-axially oriented polypropylene (BOPP) laminated with aluminium.
- **Solution**: Specific classes for `chips_packet` and `kurkure_packet` prevent classification ambiguity with standard rigid plastics, enabling proper routing for cement kiln co-processing.

### C. Co-extruded LDPE Milk Pouches & Water Sachets
- **Characteristics**: Soft, deformable film pouches (Amul, Nandini, Mother Dairy) collapse into irregular contours when discarded.
- **Solution**: Spatial data augmentation with Random Crop and Cutout enables recognition even when the pouch is flattened, crumpled, or half-folded.

### D. Clay Chai Kulhads & Poly-coated Paper Cups
- **Characteristics**: Disposable tea cups are often coated with thin polyethylene liners.
- **Solution**: Mapped to `paper_cup` for specialized pulping facilities, preventing mixed paper contamination.
