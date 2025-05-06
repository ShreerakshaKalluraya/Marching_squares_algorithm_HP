#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <string>

class ScalarField {
private:
    int width, height;
    std::vector<float> data;

public:
    ScalarField(int w, int h) : width(w), height(h), data(w * h, 0.0f) {}

    bool loadFromPGM(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return false;
        }

        std::string magic;
        file >> magic;
        if (magic != "P5" && magic != "P2") {
            std::cerr << "Not a valid PGM file: " << filename << std::endl;
            return false;
        }

        // Skip comments and whitespace
        file.get(); 
        char c = file.peek();
        while (c == '#') {
            file.ignore(10000, '\n');
            c = file.peek();
        }

        file >> width >> height;
        int maxVal;
        file >> maxVal;
        file.get(); // Skip whitespace after maxVal

        data.resize(width * height);

        // Read image data
        if (magic == "P5") { // Binary PGM
            std::vector<unsigned char> buffer(width * height);
            file.read(reinterpret_cast<char*>(buffer.data()), width * height);

            // Convert image data to floating point values (0.0-1.0)
            #pragma omp parallel for
            for (int i = 0; i < width * height; i++) {
                data[i] = buffer[i] / static_cast<float>(maxVal);
            }
        }
        else { // ASCII PGM
            for (int i = 0; i < width * height; i++) {
                int val;
                file >> val;
                data[i] = val / static_cast<float>(maxVal);
            }
        }

        return true;
    }

    // Generate a simple test field if no image is available
    void generateTestField() {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = x / static_cast<float>(width) - 0.5f;
                float ny = y / static_cast<float>(height) - 0.5f;
                
                // Simple terrain function (you can replace with more complex ones)
                float value = sin(10 * nx) * cos(10 * ny) + 
                             0.5f * sin(20 * nx) * cos(20 * ny);
                
                set(x, y, value);
            }
        }
    }
    
    float get(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height)
            return 0.0f;
        return data[y * width + x];
    }

    void set(int x, int y, float value) {
        if (x >= 0 && x < width && y >= 0 && y < height)
            data[y * width + x] = value;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
};

// Contour lines extractor using marching squares
class MarchingSquares {
private:
    const ScalarField& field;
    std::vector<float> isoLevels;
    std::vector<std::vector<std::pair<float, float>>> contourLines;

    // Lookup table for marching squares cases
    static const int CASE_TABLE[16][5];

    // Linear interpolation between two points
    std::pair<float, float> interpolate(float x1, float y1, float v1, 
                                        float x2, float y2, float v2,
                                        float isoLevel) {
        if (fabs(isoLevel - v1) < 1e-5)
            return {x1, y1};
        if (fabs(isoLevel - v2) < 1e-5)
            return {x2, y2};
        if (fabs(v1 - v2) < 1e-5)
            return {x1, y1};
        
        float t = (isoLevel - v1) / (v2 - v1);
        return {x1 + t * (x2 - x1), y1 + t * (y2 - y1)};
    }

public:
    MarchingSquares(const ScalarField& f, const std::vector<float>& levels) 
        : field(f), isoLevels(levels), contourLines(levels.size()) {}

    // Process the entire field and generate contour lines for all iso-levels
    void process() {
        int width = field.getWidth();
        int height = field.getHeight();
        
        // Process each iso-level in parallel
        #pragma omp parallel for
        for (size_t levelIdx = 0; levelIdx < isoLevels.size(); levelIdx++) {
            float isoLevel = isoLevels[levelIdx];
            
            // Thread-local storage for contour segments
            std::vector<std::pair<float, float>> levelContours;
            
            // Process all cells
            for (int y = 0; y < height - 1; y++) {
                for (int x = 0; x < width - 1; x++) {
                    // Get the values at the four corners of the cell
                    float v0 = field.get(x, y);
                    float v1 = field.get(x + 1, y);
                    float v2 = field.get(x + 1, y + 1);
                    float v3 = field.get(x, y + 1);
                    
                    // Determine the case index (0-15)
                    int caseIndex = 0;
                    if (v0 >= isoLevel) caseIndex |= 1;
                    if (v1 >= isoLevel) caseIndex |= 2;
                    if (v2 >= isoLevel) caseIndex |= 4;
                    if (v3 >= isoLevel) caseIndex |= 8;
                    
                    // Process according to the case
                    const int* caseConfig = CASE_TABLE[caseIndex];
                    for (int i = 0; caseConfig[i] != -1; i += 2) {
                        std::pair<float, float> p1, p2;
                        
                        // Determine the first point
                        switch (caseConfig[i]) {
                            case 0: p1 = interpolate(x, y, v0, x + 1, y , v1, isoLevel); break;
                            case 1: p1 = interpolate(x + 1, y, v1, x + 1, y + 1, v2, isoLevel); break;
                            case 2: p1 = interpolate(x + 1, y + 1, v2, x, y + 1, v3, isoLevel); break;
                            case 3: p1 = interpolate(x, y + 1, v3, x, y, v0, isoLevel); break;
                        }
                        
                        // Determine the second point
                        switch (caseConfig[i + 1]) {
                            case 0: p2 = interpolate(x, y, v0, x + 1, y, v1, isoLevel); break;
                            case 1: p2 = interpolate(x + 1, y, v1, x + 1, y + 1, v2, isoLevel); break;
                            case 2: p2 = interpolate(x + 1, y + 1, v2, x, y + 1, v3, isoLevel); break;
                            case 3: p2 = interpolate(x, y + 1, v3, x, y, v0, isoLevel); break;
                        }
                        
                        // Add the line segment to thread-local results
                        levelContours.push_back(p1);
                        levelContours.push_back(p2);
                    }
                }
            }
            
            #pragma omp critical
            {
                contourLines[levelIdx] = std::move(levelContours);
            }
        }
    }

    // Export all contour lines to a single SVG file
    void exportSVG(const std::string& filename) {
        int width = field.getWidth();
        int height = field.getHeight();
        
        std::ofstream out(filename);
        
        // SVG header
        out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
        out << "<svg width=\"" << width << "\" height=\"" << height 
            << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
        
        // Include a background that resembles the grayscale input image
        out << "<rect x=\"0\" y=\"0\" width=\"" << width << "\" height=\"" << height 
            << "\" fill=\"white\" />\n";
            
        // Draw a grayscale representation of the original field
        out << "<g>\n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = field.get(x, y);
                int gray = static_cast<int>((1.0f - value) * 240); // Invert so higher values are darker
                
                out << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"1\" height=\"1\" "
                    << "fill=\"rgb(" << gray << "," << gray << "," << gray << ")\" "
                    << "opacity=\"0.3\" />\n";
            }
        }
        out << "</g>\n";
            
        // Draw the grid pattern (similar to the right side of reference image)
        out << "<g stroke=\"lightgray\" stroke-width=\"0.5\">\n";
        for (int x = 0; x < width; x += 10) {
            out << "<line x1=\"" << x << "\" y1=\"0\" x2=\"" << x << "\" y2=\"" << height << "\" />\n";
        }
        for (int y = 0; y < height; y += 10) {
            out << "<line x1=\"0\" y1=\"" << y << "\" x2=\"" << width << "\" y2=\"" << y << "\" />\n";
        }
        out << "</g>\n";
        
        // Draw contour lines for each iso-level with different colors
        for (size_t i = 0; i < isoLevels.size(); i++) {
            int r = static_cast<int>(sin(0.3 * i + 0) * 127 + 127);
            int g = static_cast<int>(sin(0.3 * i + 2) * 127 + 127);
            int b = static_cast<int>(sin(0.3 * i + 4) * 127 + 127);
            
            out << "<g stroke=\"rgb(" << r << "," << g << "," << b << ")\" stroke-width=\"1\">\n";
            
            const auto& lines = contourLines[i];
            for (size_t j = 0; j < lines.size(); j += 2) {
                float x1 = lines[j].first;
                float y1 = lines[j].second;
                float x2 = lines[j + 1].first;
                float y2 = lines[j + 1].second;
                
                out << "<line x1=\"" << x1 << "\" y1=\"" << y1 
                    << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" />\n";
            }
            
            out << "</g>\n";
        }
        
        // SVG footer
        out << "</svg>\n";
        out.close();
    }
    
    // Export the contour lines as a PPM image (simple format without dependencies)
    void exportPPM(const std::string& filename) {
        int width = field.getWidth();
        int height = field.getHeight();
        
        std::ofstream out(filename, std::ios::binary);
        
        // PPM header
        out << "P6\n" << width << " " << height << "\n255\n";
        
        // Create an empty image (white background)
        std::vector<unsigned char> image(width * height * 3, 255);
        
        // Draw a grayscale representation of the original field
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = field.get(x, y);
                unsigned char gray = static_cast<unsigned char>((1.0f - value) * 230); // Higher values are darker
                
                int idx = (y * width + x) * 3;
                image[idx] = gray;
                image[idx + 1] = gray;
                image[idx + 2] = gray;
            }
        }
        
        // Draw the contour lines
        for (size_t levelIdx = 0; levelIdx < contourLines.size(); levelIdx++) {
            // Generate a color based on the iso-level
            unsigned char r = static_cast<unsigned char>(sin(0.3 * levelIdx + 0) * 127 + 127);
            unsigned char g = static_cast<unsigned char>(sin(0.3 * levelIdx + 2) * 127 + 127);
            unsigned char b = static_cast<unsigned char>(sin(0.3 * levelIdx + 4) * 127 + 127);
            
            const auto& lines = contourLines[levelIdx];
            for (size_t j = 0; j < lines.size(); j += 2) {
                // Draw the line using Bresenham's algorithm
                int x1 = static_cast<int>(lines[j].first);
                int y1 = static_cast<int>(lines[j].second);
                int x2 = static_cast<int>(lines[j + 1].first);
                int y2 = static_cast<int>(lines[j + 1].second);
                
                int dx = abs(x2 - x1);
                int dy = abs(y2 - y1);
                int sx = (x1 < x2) ? 1 : -1;
                int sy = (y1 < y2) ? 1 : -1;
                int err = dx - dy;
                
                while (true) {
                    // Check if point is within bounds
                    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                        // Set pixel color
                        int idx = (y1 * width + x1) * 3;
                        image[idx] = r;
                        image[idx + 1] = g;
                        image[idx + 2] = b;
                    }
                    
                    if (x1 == x2 && y1 == y2) break;
                    
                    int e2 = 2 * err;
                    if (e2 > -dy) {
                        err -= dy;
                        x1 += sx;
                    }
                    if (e2 < dx) {
                        err += dx;
                        y1 += sy;
                    }
                }
            }
        }
        
        // Write the image data
        out.write(reinterpret_cast<char*>(image.data()), image.size());
        out.close();
    }
};

// Lookup table for marching squares cases
const int MarchingSquares::CASE_TABLE[16][5] = {
    {-1, -1, -1, -1, -1},  // Case 0: No contour
    {0, 3, -1, -1, -1},    // Case 1
    {0, 1, -1, -1, -1},    // Case 2
    {1, 3, -1, -1, -1},    // Case 3
    {1, 2, -1, -1, -1},    // Case 4
    {0, 1, 2, 3, -1},      // Case 5 (ambiguous)
    {0, 2, -1, -1, -1},    // Case 6
    {2, 3, -1, -1, -1},    // Case 7
    {2, 3, -1, -1, -1},    // Case 8
    {0, 2, -1, -1, -1},    // Case 9
    {0, 3, 1, 2, -1},      // Case 10 (ambiguous)
    {1, 2, -1, -1, -1},    // Case 11
    {1, 3, -1, -1, -1},    // Case 12
    {0, 1, -1, -1, -1},    // Case 13
    {0, 3, -1, -1, -1},    // Case 14
    {-1, -1, -1, -1, -1}   // Case 15: No contour
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        /*
        std::cout << "Usage: " << argv[0] << " <input_file.pgm> <output_file.svg|ppm> [num_contours=10]" << std::endl;
        std::cout << "If no input file is provided or it fails to load, a test field will be generated." << std::endl;
        
        // Default parameters for test case
        const int width = 500;
        const int height = 500;
        const int numContours = 10;
        
        std::cout << "Generating test field (" << width << "x" << height << ")..." << std::endl;
        
        ScalarField field(width, height);
        field.generateTestField();
        
        // Generate iso-levels
        std::vector<float> isoLevels;
        for (int i = 0; i < numContours; i++) {
            isoLevels.push_back(-0.9f + 1.8f * i / (numContours - 1));
        }
        
        // Process the field
        MarchingSquares ms(field, isoLevels);
        ms.process();
        
        // Export the result
        ms.exportSVG("test_output.svg");
        ms.exportPPM("test_output.ppm");
        
        std::cout << "Test output saved to test_output.svg and test_output.ppm" << std::endl;
        return 0;
        */
       std::cout<<"ENter 3 inputs "<<std::endl;
       return 0;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int numContours = (argc > 3) ? std::stoi(argv[3]) : 10;
    
    // Load the input image
    ScalarField field(1, 1);  // Initial size will be updated when loading
    bool loaded = field.loadFromPGM(inputFile);
    
    if (!loaded) {
        std::cout << "Failed to load " << inputFile << ". Generating test field..." << std::endl;
        
        // Default parameters for test case
        const int width = 500;
        const int height = 500;
        
        field = ScalarField(width, height);
        field.generateTestField();
    }
    else {
        std::cout << "Loaded image: " << inputFile << " (" << field.getWidth() << "x" << field.getHeight() << ")" << std::endl;
    }
    
    // Generate iso-levels
    std::vector<float> isoLevels;
    for (int i = 0; i < numContours; i++) {
       isoLevels.push_back(static_cast<float>(i + 1) / (numContours + 1));
       //isoLevels.push_back(static_cast<float>(8) / (numContours + 1));
    }
    
    // Set number of OpenMP threads
    int numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);
    std::cout << "Using " << numThreads << " OpenMP threads" << std::endl;
    
    // Process the image
    MarchingSquares ms(field, isoLevels);
    
    std::cout << "Processing marching squares algorithm..." << std::endl;
    ms.process();
    
    // Export the result
    std::string extension = outputFile.substr(outputFile.find_last_of('.') + 1);
    if (extension == "svg") {
        std::cout << "Exporting to SVG: " << outputFile << std::endl;
        ms.exportSVG(outputFile);
    } else if (extension == "ppm") {
        std::cout << "Exporting to PPM: " << outputFile << std::endl;
        ms.exportPPM(outputFile);
    } else {
        std::cerr << "Unsupported output format: " << extension << std::endl;
        std::cerr << "Supported formats: svg, ppm" << std::endl;
        return 1;
    }
    
    std::cout << "Done!" << std::endl;
    return 0;
}