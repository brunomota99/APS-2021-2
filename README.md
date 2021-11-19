# APS-2021-2

DigiRec é um sistema de reconhecimento de digitais feito em Python, o reconhecimento 
das digitais de entrada é feito através de uma serie de transformações e no final é feito 
comparações com as digitais no diretório com o nivel de acesso apresentado. 

As transformações realizadas em sequencia: 
  - Tranformação da imagem em tons de cinza
  - Aumento de contraste utilizando histograma
  - Detecção de bordas utilizando o kernel de Sobel
  - Limearização linear da imagem
  - Segmentação da imagem para pegar apenas a digital limiarizada