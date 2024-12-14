# Vehicle count and Number Plate detection and OCR
## Vehicle detection -
  The code given computes number of vehicles entering and leaving the frame.
  A horizontal green line is used as a reference line which is used to count if vehicle is entering or leaving.
  A frame is created around the vehicles detected and their centroid is calculated to identify the location of the vehicles.

**Constraints** - 
- The camera has to be placed at a minimum of height of a tall gate.
  
**Improvements** - 
- The centroids could be used to detect the speed of the vehicles.
- The size of the frame could be used to classify vehicles into heavy and light. This could be used to later determine how fast the roads can deteriorate.

## Number Plate detection and OCR
  Various morphological operations are used to find the location of the number plate. After its extraction, more operations are performed to make the characters clearer.
  A character space is created which stores all possible characters in the number plate. Template matching is used to match the character from the number plate to the closest character found in the character space.

**Constraints** -
- The font used by the government is not available for creation of the character space.
- Only back photos of the vehicles with horizontal view is used.

**Improvements** - 
- Rules could be made for each character about their shape. These could then be used for better template matching. This will be major improvement over standard template matching.

## Usage - 
  Images and video used are stored in the 'Number Plate OCR' and 'Vehicle Detection' folder respectively. More images and videos could be used.
