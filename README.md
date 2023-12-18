# HGCLIR
Existing studies have shown that the abnormal expression of microRNAs (miRNAs) usually leads to the occurrence and development of human diseases. Identifying disease-related miRNAs contributes to study the pathogenesis of the disease at the molecular level. As traditional biological experiments are time-consuming and expensive, computational methods have been used as an effective complement to infer the potential associations between miRNAs and diseases. To this end, we develop a **H**yper**G**raph **C**ontrastive **L**earning with **I**ntegrated multi-view **R**epresentation (HGCLIR) model to discover potential miRNA-disease associations. First, hypergraph convolutional network (HGCN) is utilized to capture high-order complex relations from hypergraphs related to miRNA and disease. Then, we combine HGCN with contrastive learning to improve and enhance the embedded representation learning ability of HGCN. Next, we innovatively propose integrated representation learning to integrate the embedded representation information of multiple views for obtaining more reasonable embedding information. Finally, the integrated representation information is fed into a neural inductive matrix completion method to perform the prediction of associations between miRNAs and diseases. Experimental results on the cross-validation set and independent test set show that HGCLIR can achieve better prediction performance than other baseline models. Furthermore, the results of case studies and enrichment analysis further demonstrate the accuracy of HGCLIR and unconfirmed potential associations have biological significance.

# The workflow of HGCLIR model
![The workflow of HGCLIR model](https://github.com/Ouyang-Dong/HGCLIR/blob/master/workflow.jpg)
# Environment Requirement
The code has been tested running under Python 3.8. The required packages are as follows:
- torch == 1.12.1 (GPU version)
- numpy == 1.23.5
- pandas == 1.5.0
- scikit-learn==1.1.2

