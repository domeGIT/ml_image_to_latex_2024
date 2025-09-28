# Image To Latex

## Osnovno o projektu 
Projekat napravljen u okviru kursa Mašinsko učenje na Matematičkom fakultetu Univerziteta u Beogradu u školskoj 2024/2025. godini. Na projektu radili Dimitrije Rađenović i Tijana Jakšić.

U pitanju je implementacija modela koji na osnovu slike matematičke formule predviđa latex kod koji je ispisuje.

## Literatura:
 - Image to Latex Stanford projekat Guillaume Genthial-a https://cs231n.stanford.edu/reports/2017/pdfs/815.pdf
 - Implementacija inspirisana ovim projektom https://github.com/tuanio/image2latex/tree/main (odatle smo preuzeli vocabulary)
 - za skup podataka: https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download

## Podaci
Za skup podataka korišćen je modifikovan 100K Latex formulas dataset preuzet sa https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download.

Data folder preuređen tako da ima sledeću strukturu sadržaja:

    - im2latex .csv tabele

    - folder formula_images_processed, u kome se nalaze 
    
        - *.png slike
    
    - folder vocab, a u njemu

        - 100k_vocab.json (preuzet sa https://github.com/tuanio/image2latex/tree/main/data/vocab)

im2latex .csv tabele su modifikovane, filtrirane tako da su uklonjeni redovi duži od 172 karaktera, što je smanjilo broj formula za oko 50%.

Ovako modifikovan data folder se u obliku data.tar može preuzeti sa https://drive.google.com/file/d/11rN-xs1A2N_YRvjkCJc-cnQqlFIIRnnJ/view?usp=sharing.

## Pokretanje

Zbog ograničenih računarskih resursa ličnih računara autora, korišćen je google colab.

Pokretanje koda pomoću google colaba:

1. preuzeti data.tar sa linka https://drive.google.com/file/d/11rN-xs1A2N_YRvjkCJc-cnQqlFIIRnnJ/view?usp=sharing
2. aploudovati ga na svoj gugl drajv
3. otvoriti svesku https://colab.research.google.com/drive/1iu2iXikebtWziZMu7InBomvsmNVIjtBX#scrollTo=OPLzE5vjS6Eb
4. odabrati runtime type T4 GPU
5. pokrenuti kod