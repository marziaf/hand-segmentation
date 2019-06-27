
# Riconoscimento delle parti della mano mediante deep learning

<!-- vscode-markdown-toc -->
* 1. [Introduzione](#Introduzione)
* 2. [La rete neurale](#Lareteneurale)
	* 2.1. [Il problema della segmentazione semantica](#Ilproblemadellasegmentazionesemantica)
		* 2.1.1. [Convoluzione](#Convoluzione)
	* 2.2. [UNET](#UNET)
* 3. [Data generation](#Datageneration)
* 4. [Fonti](#Fonti)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='Introduzione'></a>Introduzione

Il progetto di segmentazione di una mano vede il suo fine in un più ampio contesto di riconoscimento gestuale, mirato al miglioramento dell'interazione uomo-macchina. Questo obiettivo richiede di passare attraverso l'identificazione delle diverse componenti della mano, per generare un input più stabile da fornire alla rete neurale per l'identificazione del gesto. In seguito, quindi, si tratterà della prima fase di questo più ampio progetto.

![Esempio rgb](images_for_presentation/desired_rgb.jpg)
![Esempio labels](images_for_presentation/desired_labels.jpg)

// TODO breve descrizione dei paragrafi a venire


##  2. <a name='Lareteneurale'></a>La rete neurale

Il problema in questione rientra in ciò che in machine learning è definito *supervised learning*. Questo consiste nell'allenare una rete neurale fornendole campioni di input e corrispondenti output attesi fino a renderla in grado di risolvere problemi analoghi ma mai visti.

L'output del caso in considerazione sarà ancora un'immagine come l'input, ma riportante le informazioni desiderate in uscita.

###  2.1. <a name='Ilproblemadellasegmentazionesemantica'></a>Il problema della segmentazione semantica

L'input della rete è stabilito: si tratta di un'immagine di una mano fornita mediante le componenti RGB e una depth map, di cui si discuterà nel dettaglio nel seguito.
L'output della rete è da stabilire, ma la scelta più naturale consiste nell'etichettare ogni singolo pixel e classificarlo assegnandogli un valore che porti con sè le informazioni desiderate. 
Si rientra quindi in un problema di classificazione, o meglio, segmentazione semantica.

In generale il problema di classificazione consiste nel ridurre un vettore di ingresso ad una informazione di dimensione minore (uno scalare). Il ridimensionamento, e con esso il concentramento dell'informazione, avviene in diversi livelli consecutivi.

![classificazione](images_for_presentation/classification_problem.png)

Questa procedura, tuttavia, non è adatta allo scopo qui trattato: infatti l'informazione a cui si vuol giungere non è di dimensione inferiore all'input e tantomeno scalare. Si ricorre quindi all'*upsampling*, che permette di ridistribuire l'informazione dopo averla concentrata.

![classificazione](images_for_presentation/deconvolution.jpg)

Prima di trattare nello specifico la rete adattata, è necessario conoscere i blocchi e le operazioni di cui si compone, che verranno quindi brevemente esposti.

####  2.1.1. <a name='Convoluzione'></a>Convoluzione

L'input dell'operazione di convoluzione consiste in una matrice tridimensionale (un volume) di taglia *n<sub>\*in</sub>\*n<sub>in</sub>\*channels* denominata anche *campo ricettivo*. A compiere la manipolazione dell'input sono *k* filtri *f\*f\*channels*.
L'output che viene generato ha dimensione *n<sub>out</sub>\*n<sub>out</sub>\*k*, che dipende dai due fattori: ![nout](images_for_presentation/nout_inline.png) dove *p* rappresenta la dimensione di *padding* della convoluzione e *s* lo *stride*.

#### Max pooling




###  2.2. <a name='UNET'></a>UNET




##  3. <a name='Datageneration'></a>Data generation
e ricordarsi di depth map


##  4. <a name='Fonti'></a>Fonti

Fonti valide

- Natural Networks and Deep Learning - Michael Nielsen

- Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation - Liang-Chieh Chen, Yukun Zhu,George Papandreou, Florian Schroff, and Hartwig Adam

- Fully Convolutional Networks for Semantic Segmentation - Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

Fonti scarse
- https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
- https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47