# Detect Tables Project

This Python project leverages computer vision techniques to extract individual cells from scanned or photographed table documents.

**Description**
* Employs edge detection methods for identifying table lines.
* Segments detected table structure into distinct cell images.
* Stores each cell as a separate image file for future processing or analysis.

**Key Classes and Functions**

* **BoxAnnotation** - A data structure to hold properties (coordinates, dimensions, class name) of detected cells.
* **TableAnalysis** - The core class:
  *  **sort_contours** - Arranges detected contours in a specified order (e.g., top-to-bottom).
  * **create_boxes_from_contours** - Generates bounding boxes around contours.
  * **get_vertical_horizontal_lines_from_image** - Extracts vertical and horizontal lines from the table image.
  * **combine_horizontal_vertical_lines** -  Combines detected lines to form a grid-like structure.
  * **store_boxes_to_column_row** - Organizes bounding boxes into tentative rows and columns.
  * **create_final_boxes** - Refines row and column arrangement, generating the final cell bounding boxes.
  * **process** - The primary function that coordinates the entire table analysis process.
  * **write_results** - Saves individual cell images to a specified directory.

**How to Run**

1. **Prerequisites:**
   *  Python 3 ([https://www.python.org/](https://www.python.org/))
   *  OpenCV (`pip install opencv-python`)
   *  NumPy (`pip install numpy`)
   *  Matplotlib (`pip install matplotlib`)

2. **Download the Project:**
   *  Clone the repository or download the project files.

3. **Place Table Images:**
   *  Put the table images you want to analyze in the `tables/` directory.

4. **Execute:**
   ```bash
   python3 <name_of_the_main_file>.py 

5. **Results:**
    Extracted cell images will be stored in the results/ directory, with subdirectories for each analyzed table image.
