
# Riconoscimento delle parti della mano mediante deep learning

<!-- vscode-markdown-toc -->
* 1. [Introduzione](#Introduzione)
* 2. [La rete neurale](#Lareteneurale)
	* 2.1. [Il problema della segmentazione semantica](#Ilproblemadellasegmentazionesemantica)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name='Introduzione'></a>Introduzione

Il progetto di segmentazione di una mano vede il suo fine in un più ampio contesto di riconoscimento gestuale, mirato al miglioramento dell'interazione uomo-macchina. Questo obiettivo richiede di passare attraverso l'identificazione delle diverse componenti della mano, per generare un input più stabile da fornire alla rete neurale per l'identificazione del gesto. In seguito, quindi, si tratterà della prima fase di questo più ampio progetto.

![Esempio rgb](images_for_presentation/desired_rgb.jpg)
![Esempio labels](images_for_presentation/desired_labels.jpg)

// TODO immagine del risultato atteso

// TODO breve descrizione dei paragrafi a venire


##  2. <a name='Lareteneurale'></a>La rete neurale

###  2.1. <a name='Ilproblemadellasegmentazionesemantica'></a>Il problema della segmentazione semantica

