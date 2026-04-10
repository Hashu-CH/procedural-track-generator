# Procedural Track Generation

Curriculum-based procedural track generation for RL agent training.
Intended use for WheeledLab on semantic navigation.

![Curriculum Progression](img/curriculum_progression.png)


## Features

- **Curriculum-based generation**: Difficulty scales from 0.0 (simple) to 1.0 (complex)
- **Two-phase system**: Phase 1 Bezier chains into Phase 2 closed loop splines
- **Feature templates**: Straight, Curve, S-Curve, Varying Curve, Chicane, and Hairpin
- **Visualization**: Built-in plotting functions for curriculum progression

### Feature Templates

Each chain segment is built from one of six feature templates, unlocked at increasing difficulty thresholds:

![Feature Showcase](img/feature_showcase.png)

### Phase Comparison

Difficulty 0.65, tracks are isolated feature segments chained in succession with intersection
checking. GEQ 0.65, tracks are non overlapping continuous loops, simulating F1 tracks.

![Phase Comparison](img/phase_comparison.png)

## Installation

### From source
```bash
git clone url
cd procedural-track-gen
pip install -e .
```

### Dependencies 
- numpy
- matplotlib
- scipy

## Usage

### Basic track generation
```python
from procedural_track_gen.curriculum import CurriculumConfig, generate_track

# Create a simple track
config = CurriculumConfig(difficulty=0.0)
grid, meta = generate_track(config=config)
print(f"Generated {meta['phase']} track with features: {meta['features']}")
```

### Running visualizations
```bash
python -c "from procedural_track_gen.viz import plot_curriculum_progression; plot_curriculum_progression()"
```

## Project Structure

- `procedural_track_gen/`: Main package
  - `__init__.py`: Package initialization
  - `bezier.py`: Spline and Bezier conversion utilities
  - `chain.py`: Chain generation logic and intersection checks
  - `curriculum.py`: Curriculum configuration and track generation
  - `features.py`: Feature definitions
  - `points.py`: Point manipulation utilities for closed tracks
  - `rasterise.py`: Rasterization functions
  - `viz.py`: Visualization helpers
