# HGCLAMIR
Existing studies have shown that the abnormal expression of microRNAs (miRNAs) usually leads to the occurrence and development of human diseases. Identifying disease-related miRNAs contributes to studying the pathogenesis of diseases at the molecular level. As traditional biological experiments are time-consuming and expensive, computational methods have been used as an effective complement to infer the potential associations between miRNAs and diseases. However, most of the existing computational methods still face three main challenges: (i) learning of high-order relations; (ii) insufficient representation learning ability; (iii) importance learning and integration of multi-view embedding representation. To this end, we developed a **H**yper**G**raph **C**ontrastive **L**earning with view-aware **A**ttention **M**echanism and **I**ntegrated multi-view **R**epresentation (HGCLAMIR) model to discover potential miRNA-disease associations. First, hypergraph convolutional network (HGCN) was utilized to capture high-order complex relations from hypergraphs related to miRNAs and diseases. Then, we combined HGCN with contrastive learning to improve and enhance the embedded representation learning ability of HGCN. Moreover, we introduced view-aware attention mechanism to adaptively weight the embedded representations of different views, thereby obtaining the importance of multi-view latent representations. Next, we innovatively proposed integrated representation learning to integrate the embedded representation information of multiple views for obtaining more reasonable embedding information. Finally, the integrated representation information was fed into a neural network-based matrix completion method to perform miRNA-disease association prediction. Experimental results on the cross-validation set and independent test set indicated that HGCLAMIR can achieve better prediction performance than other baseline models. Furthermore, the results of case studies and enrichment analysis further demonstrated the accuracy of HGCLAMIR and unconfirmed potential associations had biological significance.

# The workflow of HGCLAMIR model
![The workflow of HGCLIR model](https://github.com/Ouyang-Dong/HGCLAMIR/blob/master/workflow.jpg=600x)
# Environment Requirement
The code has been tested running under Python 3.9. The required packages are as follows:
- torch == 2.1.0 (GPU version)
- numpy == 1.26.0
- pandas == 2.1.2


