# Teaching SRU how to count

This is an implementation of SRU model, original [paper](https://arxiv.org/abs/1709.02755).
The formulas from the [paper](https://arxiv.org/pdf/2102.12459.pdf) 'When Attention Meets Fast Recurrence:
Training Language Models with Reduced Compute', were used as a reference during implementation.

The developed model predicts the next number in a 3 digit sequence.

## Implementation details
The training loss is cross entropy. Learning rate was set to 1e-3. SRU was trained for 200 epochs with a batch size of 6. 

## Dependencies
Required libraries: PyTorch and numpy. 

## Results
After trtaining the model for 200 epochs, it produces the desired result:

`For sequence: [1, 2, 3], next digit is: 4`

`For sequence: [5, 6, 7], next digit is: 8`

`For sequence: [78, 79, 80], next digit is: 81`