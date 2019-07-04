
# Riconoscimento delle parti della mano mediante deep learning

<!-- vscode-markdown-toc -->
* 1. [Introduzione](#Introduzione)
* 2. [La rete neurale](#Lareteneurale)
	* 2.1. [Il problema della segmentazione semantica](#Ilproblemadellasegmentazionesemantica)
		* 2.1.1. [Convoluzione](#Convoluzione)
		* 2.1.2. [Max pooling](#Maxpooling)
* 3. [ U-Net](#U-Net)
	* 3.1. [ Contrazione](#Contrazione)
		* 3.1.1. [ReLU](#ReLU)
	* 3.2. [Espansione](#Espansione)
* 4. [ I dati](#Idati)
	* 4.1. [Generazione dei dati sintetici](#Generazionedeidatisintetici)
	* 4.2. [Adattare i dati sintetici al modello reale](#Adattareidatisinteticialmodelloreale)
	* 4.3. [La depth map](#Ladepthmap)
* 5. [Fonti](#Fonti)

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

Prima di trattare nello specifico la rete adottata, è necessario conoscere i blocchi e le operazioni di cui si compone, che verranno quindi brevemente esposti.

####  2.1.1. <a name='Convoluzione'></a>Convoluzione

L'input dell'operazione di convoluzione consiste in una matrice tridimensionale (un volume) di taglia *n<sub>\*in</sub>\*n<sub>in</sub>\*channels*. A compiere la manipolazione dell'input sono *k* filtri *f\*f\*channels*.

La funzione dei filtri, collezioni *kernel*, è quella di scorrere lungo la matrice in input, compiendo l'operazione di convoluzione e generando in output una matrice riportante una forma di compressione locale delle informazioni originali. Ogni kernel agisce in modo indipendente sul relativo canale dell'immagine e l'output del filtro è la combinazione di questi.

L'output che viene generato ha dimensione *n<sub>out</sub>\*n<sub>out</sub>\*k*, che dipende dai due fattori: ![nout](images_for_presentation/nout_inline.png) dove *p* rappresenta la dimensione di *padding* della convoluzione e *s* lo *stride*.

####  2.1.2. <a name='Maxpooling'></a>Max pooling

L'informazione, così come fornita in input, non è esaminabile immediatamente, in quanto non è possibile avere una visione d'insieme di uno volume così ampio. L'idea di fondo è quella di ridurre le informazioni da analizzare, mantenendo solo le più importanti (nel caso del max pooling, i pixel con i valori massimi) per ogni regione.

![max_pooling](images_for_presentation/max_pooling.png)

Mediante il *pooling*, ad ogni livello i filtri diventano sempre più consci del contesto complessivo dell'immagine, in quanto questa è sempre più concentrata in poco spazio e diventa analizzabile da un singolo filtro. Grazie a questa procedura, quindi, è possibile analizzare l'immagine nella sua interezza, rendendo più chiaro *cosa* rappresenta, ma perdendo l'informazione sul *dove* i pixel si trovassero nell'input. Le procedure che vengono eseguite sono lecite nelle reti convoluzionali in virtù dell'invarianza alla traslazione delle componenti elementari di cui si compongono: non è la posizione assoluta a contare, ma quella relativa.

Si deve osservare come però, per il problema di segmentazione, questo possa sembrare controproducente: è infatti fondamentale ripristinare l'informazione spaziale.

A questo scopo interviene l'*upsampling*, per invertire la procedura di condensamento e riportare i risultati alle coordinate originali. Una tecnica che discende naturalmente dalla convoluzione è la *deconvoluzione*, o *backwards convolution*, che semplicemente è la sua inversione.

##  3. <a name='U-Net'></a> U-Net

La struttura che è stata scelta per la rete è una U-Net, sviluppata da Olaf Ronneberger per l'analisi di immagini biomediche. La rete si presta bene al problema per via della sua struttura a encoder-decoder che permette prima di comprimere e successivamente espandere il tensore in ingresso per le finalità sopra citate.

![unet](images_for_presentation/unet.jpg)

###  3.1. <a name='Contrazione'></a> Contrazione

La prima parte della rete ad essere attraversata è quella di contrazione. Qui 4 blocchi codificatori (*encoder*) si susseguono concatentati uno all'altro.

Un encoder è costituito da diversi livelli, tra cui di convoluzione, attivazione e pooling.

####  3.1.1. <a name='ReLU'></a>ReLU

Per spiegare in cosa consistono i livelli di attivazione è necessario comprendere le componenti "atomiche" di ogni rete neurale: i *neuroni*.

Un neurone è l'elemento elementare della rete, di cui i livelli sono composti. Ha lo scopo di valutare l'input pesato e a cui viene applicato un bias e a seconda del valore ottenuto decidere se attivarsi. //TODO

Una funzione di attivazione molto comune nei modelli di deep learning è la *Rectified Linear Unit*, o *ReLU*, ed è stata impiegata anche in questo contesto. Viene descritta da `f(x) = max(0,x)` e nonostante la sua semplicità è molto efficace: a differenza della altrettanto nota tangente iperbolica, l'allenamento è più efficiente e rapido, adatto a computazioni complesse.

###  3.2. <a name='Espansione'></a>Espansione

Successivamente alla contrazione, dopo aver attraversato un nodo centrale, il tensore attraversa l'ultimo ramo della rete, in cui la sua dimensione viene ripristinata a quella iniziale passando attraverso i *decoder*.

La simmetria della rete rende superflua una discussione approfondita dei decoder. Tuttavia questi ricevono in input non solo il tensore rappresentativo dell'immagine: infatti avviene anche una concatenazione tra encoder e decoder corrispondente (operanti allo stesso "livello") per inglobare anche l'informazione riguardante il contesto.

##  4. <a name='Idati'></a> I dati

L'allenamento della rete richiede una grande mole di dati in input. Questo è un ostacolo, in quanto ad ogni immagine deve essere associata una maschera rappresentante le etichette non generabile in automatico (altrimenti avremmo già la soluzione della segmentazione e la costruzione della rete sarebbe inutile).

Il numero di dati necessari per il corretto allenamento è nell'ordine delle migliaia, questo per evitare che la rete finisca in *overfitting* e per renderla flessibile ai diversi input. L'*overfitting* avviene quando i dati di allenamento della rete sono troppo pochi o troppo simili e quindi la rete impara a rispondere in modo pressochè perfetto a casi già visti, ma non è in grado di gestirne di nuovi.

Poichè non è possibile generare manualmente le maschere con le categorie per le migliaia di immagini necessarie, viene utilizzato un [generatore sintetico](http://lttm.dei.unipd.it/downloads/handposegenerator/). Questo è in grado di applicare una texture ad una mano della quale si possono controllare i movimenti.

![handgenerator](images_for_presentation/hand_generator.png)

In questo modo si riescono a generare dati illimitati con uno sforzo umano indipendente dalla dimensione del set desiderata. Tuttavia questo metodo porta degli svantaggi: i dati sintetici sono privi di rumore, variabilità e altri disturbi che invece sono presenti nelle immagini reali. 

Altro aspetto piuttosto limitante, da non sottovalutare, è la dimensione dei dati generati. Qui si è utilizzato un set di 11 gesti,per ciascuno ci sono 200 immagini 256x256 sia rgb+depth (4 canali), sia di classificazione (ad un canale), per un totale di circa 8GB. Seppur possa sembrare una quantità non troppo eccessiva, va tenuto conto che questa si deve sommare alla dimensione della rete e che queste dimensioni eccedono le normali capacità dei computer domestici. Questi problemi hanno contribuito in modo non indifferente a rallentare il progetto ed evidenziano i punti di debolezza delle reti neurali.

###  4.1. <a name='Generazionedeidatisintetici'></a>Generazione dei dati sintetici

La generazione dei dati avviene a partire da gesti predefiniti, quelli di interesse per il problema finale di riconoscimento gestuale. Questi vengono perturbati casualmente per creare più varietà possibile di immagini. 
Ad ogni posizione della mano corrispondono tre immagini generate:
- Due immagini rgb (in realtà a 4 canali, ma uno di nessun interesse): una a cui viene applicata la skin della pelle umana e una a cui corrispondono i colori rappresentanti le classi
- Un'immagine a un canale con l'informazione sulla distanza (la profondità)

Le immagini così generate, però, non sono ancora pronte per essere fornite alla rete: avviene ora la fase di assemblamento di rgb e depth in un'unico tensore e di remapping del tensore delle classi in una matrice a valori interi.

###  4.2. <a name='Adattareidatisinteticialmodelloreale'></a>Adattare i dati sintetici al modello reale

Ora i dati contengono le informazioni essenziali per il riconoscimento di gesti, ma presentano tutti delle caratteristiche comuni molto forti e innaturali.

Un problema subito evidente è l'orientamento: tutte le mani hanno il polso rivolto a sinistra. Per risolvere il problema è sufficiente applicare una rotazione casuale alle immagini, di un numero di gradi nel range 0-360 poichè sono tutte rotazioni plausbili.

Anche la centralità dell'immagine è poco naturale, ma basta applicare piccole traslazioni per ovviare al problema.

Ci sono molte altre modifiche possibili per migliorare le immagini, come riscalarle casualmente, applicare filtri per modificare la luminosità, aggiungere rumore e disturbi, cambiare colori, applicare piccole distorsioni.

###  4.3. <a name='Ladepthmap'></a>La depth map

Ulteriore vantaggio dell'uso del generatore è la generazione contestuale delle *depth map* delle immagini, ossia la rappresentazione delle informazioni sulla tridimensionalità della mano mediante un'immagine ad un canale. Questa è un'informazione ulteriore che non viene fornita dalle normali fotocamere, ma che può essere generata comunque anche nella realtà mediante sensori appositi.

![depthmap](images_for_presentation/depthmap.jpg)

## I parametri della rete



// TODO parlare di 
backpropagation
gradient descent
differenza dati reali e artificiali
parametri da scegliere
...




##  5. <a name='Fonti'></a>Fonti

Fonti valide

- Natural Networks and Deep Learning - Michael Nielsen

- Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation - Liang-Chieh Chen, Yukun Zhu,George Papandreou, Florian Schroff, and Hartwig Adam

- Fully Convolutional Networks for Semantic Segmentation - Jonathan Long, Evan Shelhamer, Trevor Darrell, UC Berkeley

- U-Net: Convolutional Networks for Biomedial Image Segmentation - Olaf Ronneberger, Philipp Fischer, and Thomas Brox

- Generalised Wasserstein Dice Score for Imbalanced Multi-class Segmentation using Holistic Convolutional Networks - Lucas Fidon,Wenqi Li, Luis C. Garcia-Peraza-Herrera, Jinendra Ekanayake, Neil Kitchen,Sébastien Ourselin, and Tom Vercauteren

- EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies - Redouane Lguensat, Member, IEEE, Miao Sun, Ronan Fablet, Senior Member, IEEE, Evan Mason, Pierre Tandeo, and Ge Chen



Fonti scarse
- https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
- https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
- https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
- https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning