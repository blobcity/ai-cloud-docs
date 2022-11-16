# ANNEXURE - COMMON CONTENT

### YAML Configuration:

Along with the above-mentioned requirements and changes, include a YAML file which contains instructions for the execution of the Usage file.

YAML file contains details regarding input format, output format, and usage file name which included above mentioned main function.

Support input and output options are JSON,doc, text, image, audio & video

| **Key** | **Required** |
| --- | --- |
| version | Optional |
| meta.name | Mandatory |
| meta.description | Mandatory |
| output.type | Mandatory |
| output.file | Conditional |
| input.type | Mandatory |
| input.file | Mandatory |
| requirements | Optional |
| performance | Optional |
| env | Mandatory |

* Conditional : Output file depends on input type and problem statement. if the problem requires saving result output then one have to include the output file key value. for example, if the problem is for face detection in the image, save the resulting bounding box image with the specified name in the YAML configuration.

**Example** :

**saved by name:** blobcity.yaml

```
version: 1
meta:
 name: Model1
 description: MODEL DESCRIPTION
input:
 type: image
 file: input.jpg
output:
 type: image
 file: output.pjpg
requirements: requirements.txt
main: usage.ipynb
performance:
 accuracy: 0.5
 f1score: 0.5 
env:
 MODEL_PATH: ./model.pkl
``` 

The relative path of the saved models should be mentioned in the ‘env’ section of the YAML configuration file. One can have n numbers of environment variables utilized in the Main function but should have a matching environment variable mentioned in the YAML file and Main function. The variable mentioned in the env section must be all Capitalized letters without any whitespace.The key *file* must be predefined by the creator/author of the model in the YAML configuration. If your model takes text input and returns text output (NLP models), enter the *file* value as null/None for input and output. While for the Image or Video Generative AI model, set the Input file value as None/null, while for the output *file* key, specify the file name with an extension. For example. output.png or output.mp4 to save the file.
