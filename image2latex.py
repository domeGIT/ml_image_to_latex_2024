import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import torchvision
import torchvision.transforms.functional as TF

import json
import pandas as pd
import numpy as np
import re
import os
import shutil



class LatexDataset(Dataset):
    """
    Formiranje pytorch skupa podataka na osnovu csv fajla cije su kolone (ime_slike, string_latex_formule)

    Nakon formiranja skupa, elementi su oblika (slika_u_obliku_tenzora, string_latex_formule)
    """
    def __init__(self, csv_path: str, transform=None):
        """
        Ulazni argumenti:
        csv_path (str): Put do CSV fajla csv fajla cije su kolone (ime_slike, string_latex_formule)
        transform (callable, optional): Opcionalna transformacija koja se primenjuje na sve slike nakon formiranja skupa
        """
        super().__init__()
        self.transform = transform
        df = pd.read_csv(csv_path)
        # promenimo kolonu image tako da ima ceo put do fajla
        df['image'] = df.image.map(lambda x: os.path.join('/content/data/formula_images_processed', f'{x}'))
        # formirajmo listu recnika (`self.walker`) gde su recnici redovi iz df
        self.walker = df.to_dict('records')

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item['formula']
        image = torchvision.io.read_image(str(item['image']))
        image = TF.rgb_to_grayscale(image, num_output_channels=1)  # (1, H, W)

        return image, formula
    

class Text():
    """
    Klasa koja enkapsulira bavljenje:
    rečnikom, tokenizacijom, i konverzijom između celobrojnih identifikatora i odgovarajućeg stringa.
    """
    def __init__(self):
        """
        Inicijalizacija: učitavanje rečnika i podešavanje pravila za tokenizaciju
        """
        # ručno podešavamo specijalne tokene
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

        # objekti koji povezuju string token sa odgovarajućim celobrojnim id-jem
        self.id2word = json.load(open("/content/data/vocab/100k_vocab.json", "r")) # lista stringova
        self.word2id = dict(zip(self.id2word, range(len(self.id2word)))) # mapa
        # regularni izraz za razbijanje latex stringa u tokene
        self.TOKENIZE_PATTERN = re.compile(
            r"(\\[a-zA-Z]+)|"           # LaTeX komande poput \frac, \sqrt
            r"((\\)*[$-/:-?{-~!\"^_`\[\]])|"  # matematički simboli
            r"(\w)|"                    # izolovana slova i brojevi
            r"(\\)"                     # pojedinačne `\`
            )
        # broj tokena
        self.n_class = len(self.id2word)

    def int2text(self, x: Tensor):
        """
        Argumenti:
            x (Tensor): 1D tenzor token ID-jeva.
        Povratna vrednost:
            str: String tokena razmaknutih razmakom, bez specijalnih (eos, sos, pad) tokena.
        """
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    def text2int(self, formula: str):
        """
        Argumenti:
            formula (string): LaTeX formula u svom string obliku
        Povratna vrednost:
            Tensor: 1D tenzor token ID-jeva
        """
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])

    def tokenize(self, formula: str):
        """
        Argumenti:
            formula (str): LaTeX formula u svom string obliku
        Povratna vrednost:
            list[str]: Lista tokena (znači lista stringova) koji odgovaraju formuli
        """
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens
    

def collate_fn(batch, text):
    """
    Funkcija koju koristi PyTorch-ev DataLoader, pri kombinovanju liste pojedinačnih uzoraka iz skupa u jedan batch.

    Podrazumevana Torcheva akcija za ovo je da samo stekuje uzorke. 
    Kako su u našem skupu formule promenljivih dužina, bilo je potrebno da napišemo posebnu collate funkciju sa odgovarajućim pad-ovanjem.

    1. Transformiše LaTeX formula string u sekvencu token ID-jeva (pomoću Text klase)
    2. Računa dužine svih sekvenci
    3. Pad-uje sekvence na dužinu najduže u batchu i dodaje sos i eos tokene
    4. Pad-uje slike na istu širinu i visinu
    5. Ovako modifikovane slike i formule stekuje u tenzore spremne za model

    Argumenti:
        batch (list[tuple]): Lista uzoraka iz dataseta. Jedan uzorak = `(image, formula_string)`
        text (Text): Instanca klase `Text`

    Povratna vrednost:
        torka:
            - images (Tensor): Float tenzor oblika `(BATCH_SIZE, CHANNELS, H, W)`
            koji sadrži ped-ovane slike
            - formulas (Tensor): Long tenzor oblika `(BATCH_SIZE, L)`
            koji sadrži ped-ovane sekvence token ID-jeva, uključujući sos i eos
            - formula_len (Tensor): Long tenzor oblika `(B,)`
            koji sadrži originalne dužine sekvenci formula - pre ped-ovanja, bez eos
    """

    formulas = [text.text2int(str(i[1])) for i in batch]
    formula_len = torch.tensor([len(f) + 1 for f in formulas], dtype=torch.long)
    formulas = pad_sequence(formulas, batch_first=True)

    batch_size = len(batch)
    sos = torch.full((batch_size, 1), text.sos_id, dtype=torch.long)
    eos = torch.full((batch_size, 1), text.eos_id, dtype=torch.long)
    formulas = torch.cat((sos, formulas, eos), dim=-1)


    images = [i[0] for i in batch]
    max_width, max_height = 0, 0
    for img in images:
        c, h, w = img.size()
        max_width = max(max_width, w)
        max_height = max(max_height, h)

    def pad_image(img):
        c, h, w = img.size()
        padding = (0, 0, max_width - w, max_height - h)
        return torchvision.transforms.functional.pad(img, padding, fill=0)

    images = [pad_image(img) for img in images]
    images = torch.stack(images).to(dtype=torch.float)

    return images, formulas, formula_len


class ConvEncoder(nn.Module):
    """
    Konvolucioni enkoder za ekstrakovanje reprezentacija atributa crno-belih slika

    Konvolucioni slojevi, ReLU aktivacije i max-pooling, potom flattening.
    """
    def __init__(self, encoder_dim: int):
        super().__init__()
        # enkoder atributa
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),    # Conv 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # downsample

            nn.Conv2d(64, 128, 3, 1, 1),  # Conv 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),           # downsample

            nn.Conv2d(128, encoder_dim, 3, 1, 1),  # Conv 3
            nn.ReLU(),
        )
        # dimenzionalnost autput vektora atributa
        self.encoder_dim = encoder_dim

    def forward(self, x: Tensor):
        """
        Forward pass konvolucionog enkodera. Input mora biti crno-bela slika.

        Argumenti:
            x (Tensor): ulazni tenzor oblika (batch_size, channels=1, width, height)

        Povratna vrednost:
            Tensor: Enkodirani tenzor atributa, oblika (batch_size, seq_len, encoder_dim)
            gde seq_len = w * h  (w i h su širina i visina slike nakon konvolucija i poolinga)
            encoder_dim je broj autput kanala u poslednjem konvolucionom sloju
        """
        encoder_out = self.feature_encoder(x)        # (bs, c, w, h)
        encoder_out = encoder_out.permute(0, 2, 3, 1) # (bs, w, h, c)
        bs, w, h, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)   # flatten spatial dims
        return encoder_out

class Attention(nn.Module):
    def __init__(self, enoder_dim: int = 512, decoder_dim: int = 512, attention_dim: int = 512):
        super().__init__()

        """
        Racunamo kontekst vektor na osnovu sledecih jednacina
        e = tanh((Wₕhₜ₋₁ + bₕ) + (WᵥV + bᵥ))
        αₜ = Softmax(Wₐ·e + bₐ)
        cₜ = ∑ᵢ αₜⁱ vᵢ, where vᵢ ∈ V
        """
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim, bias=False) # W_h * h_{t-1}
        self.encoder_attention = nn.Linear(enoder_dim, attention_dim, bias=False) # W_V * V
        self.attention = nn.Linear(attention_dim, 1, bias=False)      # W_a * attn

        # Softmax će pretvoriti sirove rezultate u raspodelu verovatnoće (težine pažnje).
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h: Tensor, V: Tensor):
        """
        Izračunaj kontekst vektor tako što pažljivo posmatraš najrelevantnije delove slike.

        Argumenti:
            h: Prethodno skriveno stanje LSTM dekodera. Oblik: (batch_size, decoder_dim)
            V: Mapa karakteristika. Oblik: (batch_size, w * h, encoder_dim)

        Povratna vrednost:
            context (Tensor): Vektor koji iz mapa karakteristike izvlaci relevantne podatke za generisanje sledeceg karaktera.
                            Oblik: (batch_size, decoder_dime)
        """


        attn_1 = self.decoder_attention(h) #(b, decoder_dim) -> (b, attention_dim)
        attn_2 = self.encoder_attention(V) #(b, w*h, enoder_dim) -> (b, w*h, attention_dim)

        attention= self.attention(torch.tanh(attn_1.unsqueeze(1) + attn_2)).squeeze(2)
        # attn_1.unsqueeze(1): (b, 1, attention_dim)
        # attn_2: (b, w*h, attention_dim)
        # tanh(): (b, w*h, attention_dim)
        # attention: (b, w*h, 1) -> squeeze(2) -> (b, w*h)

        alpha = self.softmax(attention)


        context = (alpha.unsqueeze(2) * V).sum(dim=1)
        # alpha.unsqueeze(2): (b, w*h, 1)
        # V: (b, w*h, enoder_dim)
        # product: (b, w*h, enoder_dim)
        # context: (b, enoder_dim)
        return context

class Decoder(nn.Module):
    def __init__(self,n_class: int,embedding_dim: int = 80,encoder_dim: int = 512,decoder_dim: int = 512,attention_dim: int = 512,
        num_layers: int = 1,dropout: float = 0.1,bidirectional: bool = False,sos_id: int = 1,eos_id: int = 2):
        super().__init__()

        """
        Implementacija dekodera za Image-to-Latex model.
        Koristi LSTM ćeliju i Luong pažnju da generiše LaTeX simbole korak po korak.
        cₜ = Attention(hₜ₋₁, V)
        eₜ = Embedding(yₜ)
        (oₜ, hₜ) = LSTM(hₜ₋₁, [cₜ, eₜ])
        p(yₜ₊₁ | y₁, ..., yₜ) = Softmax(Wₒ · oₜ + bₒ)

        """

        self.sos_id = sos_id
        self.eos_id = eos_id

        # Embedding layer konvertuje token ID u vektor
        self.embedding = nn.Embedding(n_class, embedding_dim)  # (vocab_size, embedding_dim)

        # Instanca mehanizma pažnje
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # Veličina enkodera -> veličina pažnje

        # Linearni sloj za spajanje embeddinga i konteksta pažnje
        self.concat = nn.Linear(embedding_dim + encoder_dim, decoder_dim)  # (embedding_dim + encoder_dim) -> decoder_dim

        # Prvi LSTM sloj
        self.rnn = nn.LSTM(
            decoder_dim,
            decoder_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Dropout za regularizaciju
        self.dropout = nn.Dropout(dropout)

        # Drugi LSTM sloj za dublji model
        self.rnn2 = nn.LSTM(
            decoder_dim,
            decoder_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Izlazni sloj koji preslikuje u prostor rečnika
        self.out = nn.Linear(decoder_dim, n_class)  # (decoder_dim) -> (n_class)

        # LogSoftmax za stabilnost prilikom računanja gubitka
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # Inicijalizacija težina
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, nn.Embedding):
            nn.init.orthogonal_(layer.weight)
        elif isinstance(layer, nn.LSTM):
            for name, param in self.rnn.named_parameters():
                if name.startswith("weight"):
                    nn.init.orthogonal_(param)

    def forward(self, y, encoder_out=None, hidden_state=None):
        """
        Generiše sledeći token na osnovu trenutnog stanja i izlaza enkodera.

        Argumenti:
            y: Ulazni tokeni. Oblik: (batch_size, target_len)
            encoder_out: Izlaz enkodera (V). Oblik: (batch_size, encoder_dim, w', h')
            hidden_state: Prethodno skriveno stanje (h, c). Oblik: (num_layers * num_directions, batch_size, decoder_dim)

        Povratna vrednost:
            out: Log-verovatnoće za sledeći token. Oblik: (batch_size, 1, n_class)
            hidden_state: Ažurirano skriveno stanje.
        """

        h, c = hidden_state  # (b, decoder_dim), (b, decoder_dim)

        embed = self.embedding(y)  # (b, seq_len, embedding_dim)
        attention_context = self.attention(h, encoder_out)  # (b, encoder_dim)

        rnn_input = torch.cat([embed[:, -1], attention_context], dim=1)  # (b, embedding_dim + encoder_dim)
        rnn_input = self.concat(rnn_input)  # (b, decoder_dim)

        rnn_input = rnn_input.unsqueeze(1)  # (b, 1, decoder_dim)
        hidden_state = (h.unsqueeze(0), c.unsqueeze(0))  # (1, b, decoder_dim), (1, b, decoder_dim)

        out, hidden_state = self.rnn(rnn_input, hidden_state)  # out: (b, 1, decoder_dim)

        out = self.dropout(out)  # (b, 1, decoder_dim)

        out, hidden_state = self.rnn2(out, hidden_state)  # out: (b, 1, decoder_dim)
        out = self.logsoftmax(self.out(out))  # (b, 1, n_class)

        h, c = hidden_state
        return out, (h.squeeze(0), c.squeeze(0))  # Squeeze dimenziju slojeva

class Image2LatexModel(nn.Module):
    """

    """
    def __init__(self,n_class: int,embedding_dim: int = 80,encoder_dim: int = 512,decoder_dim: int = 512,attention_dim: int = 512,
        num_layers: int = 1,dropout: float = 0.1,bidirectional: bool = False,text: Text = None, beam_width: int = 5, sos_id: int = 1,eos_id: int = 2, decode_type: str = "greedy"):
        super().__init__()
        self.encoder = ConvEncoder(encoder_dim=encoder_dim)
        self.decoder = Decoder(n_class=n_class,embedding_dim=embedding_dim,encoder_dim=encoder_dim,decoder_dim=decoder_dim,attention_dim=attention_dim,num_layers=num_layers,dropout=dropout,bidirectional=bidirectional,sos_id=sos_id,eos_id=eos_id)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.n_class = n_class
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width
        self.encoder = ConvEncoder(encoder_dim=encoder_dim)
        self.criterion = nn.CrossEntropyLoss()

    # TODO: why do we init the hidden state like this and not just 0?
    def init_decoder_hidden_state(self, V: Tensor):
        """
        Inicijalizuje skriveno stanje dekodera na osnovu autputa enkodera
        Argumenti:
            V (Tensor): autput enkodera, oblika (batch_size, seq_len=w*h, encoder_dim).
        Povratna vrednost:
            (h, c): Dvojka tenzora, oba oblika (batch_size, decoder_dim),
            koji predstavljaju inicijalno skriveno i stanje ćelije za dekoder.
        """
        encoder_mean = V.mean(dim=1)
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c

    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        """
        Propagacija unapred
        Argumenti:
            x (Tensor): Ulazna slika kao tenzor oblika (batch_size, channels, H, W).
            y (Tensor): Vrednosti ID-jeva koji odgovaraju stvarnim tokena,
            tenzor oblika (batch_size, seq_len).
            y_len (Tensor): Dužine stvarnih sekvenci pre pad-ovanja,
            tenzor oblika (batch_size,).

        Povratna vrednost:
            Tensor: Logiti za poczicije tokena,
            tenzor oblika (batch_size, seq_len, vocab_size),
        """
        encoder_out = self.encoder(x)

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        predictions = []

        for t in range(y_len.max().item()):
            dec_input = y[:, t].unsqueeze(1)
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def decode_greedy(self, x: Tensor, max_length: int = 150):
        """
        Greedy dekoding: odabrati najverovatniji token u svakom koraku.

        Argumenti:
            x (Tensor): Ulazna slika kao tenzor oblika (batch_size, channels, H, W).
            max_length (int, optional): max dužina povrstne liste predviđenih
                ID-jeva tokena. Podrazumevano: 150.

        Povratna vrednost:
            List[int]: Sekvenca predviđenih ID-jeva tokena (dužine <= max_length).
        """
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)
        device = encoder_out.device

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        y = torch.tensor([self.decoder.sos_id], device=device).view(bs, -1)

        predictions = []
        for t in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)

            k = out.argmax().item()
            predictions.append(k)

            y = torch.tensor([k], device=device).view(bs, -1)
        return predictions

    def decode_beam_search(self, x, max_length=150):
      """
      Dekodiranje pomoću beam search-a: u svakom koraku čuva najboljih k kandidata za sekvence.

      Argumenti:
            x (Tensor):Ulazna slika kao tenzor oblika (1, channels, H, W)
                        #!!TODO: obezbediti batch dekodiranje, da x može da bude oblika
                        (batch_size, channels, H, W)
            max_length (int, optional): max dužina povrstne liste predviđenih
                ID-jeva tokena. Podrazumevano: 150.

        Returns:
            List[int]:  Sekvenca predviđenih ID-jeva tokena, sa najvećim beam skorom
      """
      encoder_out = self.encoder(x)
      bs = encoder_out.size(0)  # 1

      hidden_state = self.init_decoder_hidden_state(encoder_out)

      list_candidate = [
          ([self.decoder.sos_id], hidden_state, 0)
      ]  # (input, hidden_state, log_prob)
      for t in range(max_length):
          new_candidates = []
          for inp, state, log_prob in list_candidate:
              y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)
              out, hidden_state = self.decoder(y, encoder_out, state)

              topk = out.topk(self.beam_width)
              new_log_prob = topk.values.view(-1).tolist()
              new_idx = topk.indices.view(-1).tolist()
              for val, idx in zip(new_log_prob, new_idx):
                  new_inp = inp + [idx]
                  new_candidates.append((new_inp, hidden_state, log_prob + val))

          new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
          list_candidate = new_candidates[: self.beam_width]

      return list_candidate[0][0]

    def decode_greedy_batch(self, x: Tensor, max_length: int = 150):

        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)
        device = encoder_out.device

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        y = torch.full((bs, 1), self.decoder.sos_id, dtype=torch.long, device=device)

        sequences = [[self.decoder.sos_id] for _ in range(bs)]  # store sequences per image
        finished = [False] * bs

        for t in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)
            preds = out.argmax(dim=-1) # (batch_size, 1)

            for i in range(bs):
                if not finished[i]:
                    token_id = preds[i].item()
                    sequences[i].append(token_id)
                    if token_id == self.decoder.eos_id:
                        finished[i] = True

            y = preds  # next input

            if all(finished):
                break
        return sequences

    def decode(self, x: Tensor, max_length: int = 150):
        """
        Decode funkcija, u zavisnosti od `self.decode_type` podržava
        greedy ili beam search enkoding.
        """
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
        elif self.decode_type == "beamsearch":
            predict = self.decode_beam_search(x, max_length)
        return predict

    def compute_loss(self, outputs, formulas_out):
        """
        Funkcija računa cross entropy loss između predviđanja modela i stvarnih vrednosti.

        Argumenti:
            outputs (Tensor): Predviđanja modela, tenzor oblika
                (batch_size, seq_len, vocab_size).
            formulas_out (Tensor): Stvarna vrednost ID-jeva tokena, oblika
                (batch_size, seq_len).

        Povratna vrednost:
            Tensor: Skalarna povratna vrednost loss funkcije.
        """
        bs, t, _ = outputs.size()
        return self.criterion(
            outputs.reshape(bs * t, -1),   # flatten predictions
            formulas_out.reshape(-1)       # flatten targets
        )

def exact_match(pred_list, truth_list):
    """
    Računa EM skor između predviđenih i stvarnih vrednosti sekvenci.

    Argumenti:
        pred_list (List[List[str]]): batch predviđenih sekvenci tokena
        truth_list (List[List[str]]): batch stvarnih sekvenci tokena

    Povratna vrednost:
        Tensor: skalarni tenzor koji sadrži srednju vrednost EM skorova u batchu.
        vrednosti EM skorova u batchu su:
            - 1, ako su predviđena i stvarna sekvenca identične
            - 0, inače
    """
    em_scores = []
    for pred, truth in zip(pred_list, truth_list):
        len_pred = len(pred)
        len_truth = len(truth)
        max_len = max(len_pred, len_truth)

        # Pad both sequences to the same length
        padded_pred = pred + [""] * (max_len - len_pred)
        padded_truth = truth + [""] * (max_len - len_truth)

        # Calculate EM for this single pair
        em = (np.array(padded_pred) == np.array(padded_truth)).all()
        em_scores.append(em)

    # Return the mean EM score for the entire batch
    return torch.tensor(np.mean(em_scores))

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bind_gpu(data):
    device = get_device()
    if isinstance(data, (list, tuple)):
        return [bind_gpu(data_elem) for data_elem in data]
    else:
        return data.to(device, non_blocking=True)