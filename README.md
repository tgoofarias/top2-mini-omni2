
# Trabalho de Tópicos Avançados em Computação II

# Equipe

 * Manuela Cristina Pereira Bastos
 * Nickolas Javier Santos Livero
 * Tiago Farias Barbosa

### Análise do modelo Mini-Omni2 Any-to-Any

Para evitar confusão entre os códigos já implementados pelo criador e código feito pela equipe por favor ler este README

# Código de análise do modelo

 * test.py

No arquivo test.py possui o código responsável por importar o modelo e preparar os inputs para análise, também responsável por tratar os outputs como salvá-los em arquivos de aúdios.

 * Pasta tests

Nesta pasta armazena os exemplos, tanto inputs quanto outputs do modelo.

Foram realizados dois tipos de teste:

    - Apenas com aúdio
    - Com Aúdio e Vídeo

**OBSERVAÇÃO: cada pasta possui um output_audio.wav e um response.txt que correspondme as saídas do modelo**

 * tests/audios

Nesta pasta se encontra os testes do modulo com audio, tanto good_audio com um áudio em um bom cenário e um bad_audio com um aúdio em um cenário ruim.

* tests/videos

Nessa pasta possui exemplos de vídeo nas pastas **dog**, **car**, **rio**. Neles se encontra os resultados dos testes com vídeo e aúdio.