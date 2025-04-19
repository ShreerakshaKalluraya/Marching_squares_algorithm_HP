# Marching Squares Algorithm Implementation

This project implements the Marching Squares algorithm for generating contour lines from scalar fields or grayscale images. It uses OpenMP for parallel processing to improve performance on multi-core systems.

## Overview

The Marching Squares algorithm is a computer graphics technique used to create contour lines or isolines that represent constant values in a 2D scalar field. It's commonly used in:

- Topographic maps
- Weather visualization (isobars, isotherms)
- Medical imaging
- Scientific visualization

## Features

- Load grayscale images in PGM format as input scalar fields
- Generate test scalar fields with mathematical functions
- Parallel processing using OpenMP
- Multiple contour levels
- Export results as SVG or PPM files
- Configurable number of contour lines

## Requirements

- C++ compiler with C++11 support
- OpenMP support

## Compilation

To compile the program, use:

```bash
g++ -fopenmp marching_squares.cpp -o marching_squares
```

## Usage

### Basic Usage

```bash
./marching_squares [input_file.pgm] [output_file.svg|ppm] [num_contours]
```

### Parameters

- `input_file.pgm`: Input grayscale image in PGM format
- `output_file.svg|ppm`: Output file (SVG or PPM format)
- `num_contours`: Number of contour levels (optional, default is 10)

### Examples

Generate contours from an input image:
```bash
./marching_squares input.pgm output.svg 10
```





## Input Format

The program accepts PGM (Portable Gray Map) images as input. PGM is a simple grayscale image format.

## Output Format

### SVG

The SVG output includes:
- A background showing the original grayscale data
- A grid pattern for reference
- Colored contour lines for each iso-level

SVG files can be viewed in any web browser or vector graphics editor.



## Algorithm

The Marching Squares algorithm works by:

1. Dividing the scalar field into a grid of cells
2. For each cell, determining which corners are above the threshold (iso-level)
3. Using a lookup table to determine how the contour line passes through the cell
4. Interpolating the exact position of contour points along cell edges
5. Connecting these points to form contour lines

## Implementation Details

- `ScalarField` class: Represents a 2D scalar field and handles loading/generating data
- `MarchingSquares` class: Implements the algorithm and exports results
- Lookup table: Contains the 16 possible cases for how a contour can cross a cell
- Linear interpolation: Used to find precise contour positions between grid points
- OpenMP parallelization: Process multiple contour levels and cells concurrently

## Performance Tips

- Adjust the number of OpenMP threads based on your system's CPU cores
- For large images, consider reducing resolution before processing
- Processing time scales with both image size and number of contour levels


