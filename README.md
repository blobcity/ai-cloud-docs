# ANNEXURE - COMMON CONTENT

### YAML Configuration:

Along with the above-mentioned requirements and changes, include a YAML file which contains instructions for the execution of the Usage file.

YAML file contains details regarding input format, output format, and usage file name which included above mentioned main function.

Support input and output options are JSON, text, image, audio & video

| **Key** | **Required** |
| --- | --- |
| version | Optional |
| meta.name | Mandatory |
| meta.description | Mandatory |
| output.type | Mandatory |
| input.type | Mandatory |
| requirements | Optional |
| performance | Optional |
| env | Mandatory |

**Example** :

**saved by name:** blobcity.yaml

```
version: 1
meta:
 name: Model1
 description: MODEL DESCRIPTION
input:
 type: image
output:
 type: image
requirements: requirements.txt
main: usage.ipynb
performance:
 accuracy: 0.5
 f1score: 0.5 
env:
 MODEL_PATH: ./model.pkl
``` 

The relative path of the saved models should be mentioned in the ‘env’ section of the YAML configuration file. One can have n numbers of environment variables utilized in the Main function but should have a matching environment variable mentioned in the YAML file and Main function. The variable mentioned in the env section must be all Capitalized letters without any whitespace.
