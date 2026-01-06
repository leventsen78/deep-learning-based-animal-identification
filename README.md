# deep-learning-based-animal-identification

Levent Şen

Summary: Flowchart

The process works as follows:

Input: Get the image.jpg path from the user.

Preprocessing: Open image -> Convert to RGB -> Resize to 160x160 -> Scale numbers between 0-1 -> Add dimension.

Model: Send processed data to the .keras file.

Output: Get raw probabilities from the model (e.g., Index 5 = 90%).

Mapping: What is Index 5? -> Check JSON ("tiger").

Translation: What is "tiger" in Turkish? -> Check dictionary ("Kaplan").

Result: Print "KAPLAN (tiger) 90%" to the screen.
