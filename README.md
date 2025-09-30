# Image To Latex

## Osnovno o projektu 
Projekat napravljen u okviru kursa Mašinsko učenje na Matematičkom fakultetu Univerziteta u Beogradu u školskoj 2024/2025. godini. Na projektu radili Dimitrije Rađenović i Tijana Jakšić.

U pitanju je implementacija modela koji na osnovu slike matematičke formule predviđa LaTeX kod koji je ispisuje.

## Literatura:
 - Image to Latex Stanford projekat Guillaume Genthial-a https://cs231n.stanford.edu/reports/2017/pdfs/815.pdf
 - Implementacija inspirisana Guillaume-ovim projektom https://github.com/tuanio/image2latex/tree/main (odatle smo preuzeli gotov vocabulary)
 - Skup podataka: https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download

## Podaci
Za skup podataka korišćen je modifikovan 100K Latex formulas dataset preuzet sa .

### Sadržaj Data foldera:

- im2latex CSV tabele unapred podeljene na trening, validaciju i test

    - folder formula_images_processed, u kome se nalaze 
    
        - preprocesirane PNG slike renderovanih LaTeX formula
    
- folder vocab, a u njemu

    - 100k_vocab.json (preuzet <a href="https://github.com/tuanio/image2latex/tree/main/data/vocab"> odavde </a>)

im2latex CSV tabele i PNG slike su preuzete iz standardnog <a href="https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k?resource=download">100K LaTeX formulas dataseta</a>. CSV tabele su modifikovane filtriranjem. Isključeni su svi redovi koji sadeže formule duže od 162 karaktera, što je smanjilo broj formula za oko 50%.

Ovako modifikovan data folder se u obliku data.tar je okačen u projekat na githab pomoću LFS.

## Paketi potrebni za pokretanje projekta
- torch 
- torchvision
- matplotlib
- pandas
- numpy
- os
- shutil (potrebno za pokretanje u Google Colab-u)
- re
- json
- nltk

## Pokretanje

Zbog ograničenih računarskih resursa ličnih računara autora, korišćen je google colab, i sveska je u potpunosti prolagođena da sarađuje sa tim okruženjem.

Pokretanje koda pomoću Google Colab-a:

1. Otići na <a href="https://colab.research.google.com/">google colab</a>
2. U prozoru "open notebook" odabrati opciju GitHub
3. U polje za pretragu uneti https://github.com/domeGIT/ml_image_to_latex_2024
4. Odabrati svesku za pokretanje
5. Ukoliko su u pitanju sveske za trening ili test, podesiti da runtime bude grafička kartica (Runtime > Change runtime type)
6. Pokrenuti kod
7. U koliko se pojavi greška da `nltk` biblioteka ne postoji, otvoriti terminal i pokrenuti `pip install nltk`. Ostali paketi su podrazumevano uključeni u Google Colab.
