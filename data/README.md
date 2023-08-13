# Dataset of OJ
## raw_data
### ProgramData 
Because using pycparser will invalidate the tokenizer of some of the original code, the following modifications have been made
- In the code more than a '/' directly delete
- The type definition '#define' is used, changing '#define' to the corresponding type declaration

|       | Examples | Program Tasks |
| ----- | :-------: |:-------: |
| OJ-Data |  52,000  | 104 |

## `read_data.py`
### Data reading and partitioning
> function `split_raw_dataset()`
- Objective: to use the same training set, verification set and test set for different models
- Partition ratio  `3.2:0.8:1`=〉33280:8320:10400 
- Storage location: Stored in pkl file in train/test/dev，
- Stored in the form of a dictionary
  - key value of dictionary：`id`,`code`和`label`
  - The item corresponding to each key is in the form of a list
