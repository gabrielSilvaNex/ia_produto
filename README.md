<p align="center">
  <a href="" rel="noopener">
</p>

<h3 align="center">IA para reconhecimento de produtos</h3>

---

<p align="center"> Este read me tem como objetivo descrever como funciona para criar uma ia capaz de reconhecer logos em uma imagem.
    <br> 
</p>

## 📝 Conteúdo

- [Sobre](#Sobre)
- [Bibliotecas](#bibliotecas)
- [Modelos](#modelos)
- [prints](#prints)
- [Treinamento](#treinamento)
- [Teste](#teste)

## 🧐 Sobre <a name = "Sobre"></a>

Este projeto implementa um sistema de reconhecimento e contagem de produtos. O sistema utiliza uma abordagem em três camadas com detecção de área de exposição, detecção de produtos e reconhecimento específico do produto através de embeddings de similaridade.

## 🏁 Bibliotecas <a name = "bibliotecas"></a>

O projeto utiliza as seguintes bibliotecas principais:

- TensorFlow/Keras para deep learning e metric learning
- Ultralytics YOLO para detecção de objetos

Todas as dependências estão listadas no arquivo `requirements.txt`.

## 🔧 Passo 2 -> Modelos para reconhecimento de tipo de exposição e produto <a name = "modelos"></a>

De maneira geral, este modelo de reconhecimento de produtos funciona em 3 camadas:

1. Ele passa primeiro por um reconhecimento de tipo de exposição, para não analisarmos produtos que estejam fora da exposição.
2. Ele passa por um reconhecimento de "o que é" um produto, e assim é recortada essa imagem de produto para ir para o proximo passo.
3. Finalmente ele passa pelo reconhecimento do produto em si, e cada produto da imagem tem a sua imagem recortada e analisada pelo modelo de reconhecimento de produtos.

### Modelos de Exposição e produto

Para o primeiro e segundo passo será necessário o treinamento de uma IA parecida com a IA de marca, porém para reconhecer tipos de exposição e uma outra para reconhecer o que é um produto (essa terá só uma classe mesmo).

E após esses dois modelos prontos podemos seguir para a criação do modelo de reconhecimento de produtos.

## 🎈 Passo 3 -> Prints dos produtos <a name="prints"></a>

Será necessário 6 prints de cada produto para fazer o treinamento (quando mais, melhor).
Separe os prints em pasta, onde o nome da pasta será o nome da classe desse produto em expecifico.

## 🚀 Passo 4 -> Treinamento <a name = "treinamento"></a>

O processo de treinamento é dividido em duas etapas principais:

### 1. Treinamento do Modelo de Embeddings (train_embedder.py)

Este script treina um modelo de deep learning para gerar embeddings de produtos usando metric learning

### 2. Geração da Biblioteca de Embeddings (gerar_biblioteca_embeddings.py)

Este script processa as imagens de referência para criar uma biblioteca de comparação:

- Carrega o modelo treinado
- Processa todas as imagens da pasta com as imagens
- Gera embeddings normalizados
- Salva biblioteca em formato NPZ com:
  - Embeddings das imagens
  - IDs dos produtos correspondentes

O resultado final são dois arquivos essenciais:

- `models/product_embedder.keras`: Modelo treinado
- `models/biblioteca_produtos.npz`: Biblioteca de embeddings

## ⛏️ Testes <a name = "teste"></a>

O script `contador_de_produtos.py` implementa o pipeline completo de detecção e contagem de produtos.

### Funcionamento

1. **Detecção de Área**

   - Usa `area_detector.pt` para identificar áreas de exposição
   - Aplica threshold de confiança configurável
   - Marca áreas do tipo de exposição na imagem

2. **Detecção de Produtos**

   - Usa `product_detector.pt` em cada área detectada
   - Recorta produtos individuais
   - Aplica threshold de confiança específico

3. **Reconhecimento de Produtos**

   - Gera embedding para cada produto detectado
   - Compara com biblioteca de referência
   - Identifica produto mais similar

4. **Visualização e Contagem**
   - Produtos reconhecidos: verde
   - Produtos desconhecidos: vermelho
   - Mostra nome e similaridade
   - Gera contagem total por produto

### Uso

```bash
python contador_de_produtos.py --imagem caminho/para/imagem.jpg
```

### Saída

- Imagem anotada salva como `resultado.jpg`
- Relatório de contagem no terminal
- Visualização de bounding boxes e labels
