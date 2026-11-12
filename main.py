from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()


"""
LCEL (Langchain Expression Language): padrão langchain para criar pipelines de 
dados conectados a LLMs usando conceito de composição de Funções
"""

def main():
    print("Hello from langchain-course!")
    information = """
El Niño é uma das duas fases do El Niño-Oscilação do Sul (ENOS), sendo o seu "oposto" o La Niña. É um fenômeno climático global que surge de variações nos ventos e nas temperaturas da superfície do mar sobre o Oceano Pacífico tropical. Essas variações têm um padrão irregular, com alguma semelhança com ciclos, mas não sendo previsíveis a longo prazo. Ele afeta o clima de grande parte dos trópicos e subtrópicos e tem ligações (teleconexões) com regiões de latitudes mais altas do mundo, sendo marcado por aquecimento da temperatura da superfície do mar - a fase de resfriamento é conhecido comoLa Ninã. A Oscilação Sul é a oscilação atmosférica que acompanha a mudança da temperatura do mar.

O fenômeno El Niño está associado a uma pressão atmosférica acima do normal ao nível do mar sobre a Indonésia, Austrália e através do Oceano Índico até o Atlântico. La Niña tem aproximadamente o padrão inverso: alta pressão sobre o Pacífico central e oriental e menor pressão em grande parte do resto dos trópicos e subtrópicos.[2][3] Os dois fenômenos duram cerca de um ano cada e ocorrem normalmente a cada dois a sete anos com intensidade variável, com períodos neutros de menor intensidade intercalados.[4] Os eventos El Niño podem ser mais intensos, mas os eventos La Niña podem se repetir e durar mais.

Um mecanismo fundamental do ENOS é o feedback de Bjerknes (nomeado em homenagem a Jacob Bjerknes em 1969), no qual as mudanças atmosféricas alteram as temperaturas do mar, que por sua vez alteram os ventos atmosféricos em um feedback positivo. Ventos alísios mais fracos do leste resultam em uma onda de águas superficiais quentes para o leste e redução da ressurgência oceânica no equador. Por sua vez, isso leva a temperaturas mais altas na superfície do mar (chamado El Niño), uma circulação de Walker mais fraca (uma circulação leste-oeste na atmosfera) e ventos alísios ainda mais fracos. No final das contas, as águas quentes do Pacífico tropical ocidental estão esgotadas o suficiente para que as condições retornem ao normal. Os mecanismos exatos que causam a oscilação não são claros e estão sendo estudados.

Cada país que monitoriza o ENOS tem um limiar diferente para o que constitui um evento El Niño ou La Niña, que é adaptado aos seus interesses específicos.[5] El Niño e La Niña afetam o clima global e perturbam os padrões meteorológicos normais, o que pode levar a tempestades intensas em alguns lugares e a secas em outros.[6][7] Os eventos El Niño causam picos de curto prazo (aproximadamente 1 ano de duração) na temperatura média global da superfície, enquanto os eventos La Niña causam resfriamento de curto prazo da superfície.[8] Portanto, a frequência relativa dos eventos El Niño em comparação com os eventos La Niña pode afetar as tendências da temperatura global em escalas de tempo de cerca de dez anos.[9] Os países mais afetados pelo ENOS são os países em desenvolvimento que fazem fronteira com o Oceano Pacífico e dependem da agricultura e da pesca.

Na ciência das mudanças climáticas, o ENOS é conhecido como um dos fenômenos de variabilidade climática interna.[10]:23As tendências futuras do ENOS devido às alterações climáticas são incertas,[11] embora as alterações climáticas agravem os efeitos das secas e das cheias. O Sexto Relatório de Avaliação do IPCC resumiu o conhecimento científico em 2021 para o futuro do ENOS da seguinte forma: "A longo prazo, é muito provável que a variação da precipitação relacionada com o El Niño-Oscilação do Sul aumente".[10]:113 O consenso científico é também que “é muito provável que a variabilidade da precipitação relacionada com as alterações na força e na extensão espacial das teleconexões ENOS conduza a alterações significativas à escala regional”.[10]:114

Definição e terminologia

Séries temporais do Índice de Oscilação Sul de 1876 a 2025. A Oscilação Sul é o componente atmosférico do El Niño. Este componente é uma oscilação na pressão atmosférica da superfície entre as águas tropicais do leste e do oeste do Oceano Pacífico.
A Oscilação El Niño-Sul é um fenômeno climático único que oscila periodicamente entre três fases: Neutra, La Niña ou El Niño.[12] La Niña e El Niño são fases opostas na oscilação que são consideradas como ocorrendo quando condições oceânicas e atmosféricas específicas são atingidas ou excedidas.[12]

Uma menção registrada do termo "El Niño" ("O Menino" em espanhol) para se referir ao clima ocorreu em 1892, quando o Capitão Camilo Carrillo disse ao congresso da sociedade geográfica em Lima que os marinheiros peruanos batizaram a corrente quente que flui para o sul de "El Niño" porque ela era mais perceptível perto do Natal.[13] Embora as sociedades pré-colombianas estivessem certamente cientes do fenômeno, os nomes indígenas para ele se perderam na história.[14]

O termo El Niño em maiúsculas refere-se ao menino Jesus, porque o aquecimento periódico no Pacífico perto da América do Sul é geralmente notado perto do Natal.[15]

Originalmente, o termo El Niño aplicava-se a uma corrente oceânica anual fraca e quente que corria para sul ao longo da costa do Peru e do Equador.[16] No entanto, com o tempo, o termo evoluiu e agora se refere à fase quente e negativa do fenômeno El Niño-Oscilação Sul (ENOS). A frase original, El Niño de Navidad, surgiu há séculos, quando pescadores peruanos deram ao fenômeno climático o nome do Cristo recém-nascido.[17][18]

La Niña ("A Menina" em espanhol) é a contrapartida mais fria do El Niño, como parte do padrão climático ENOS mais amplo. No passado, também era chamado de anti-El Niño[19] e El Viejo, que significa "o velho".[20]

Uma fase negativa existe quando a pressão atmosférica sobre a Indonésia e o Pacífico ocidental é anormalmente alta e a pressão sobre o Pacífico oriental é anormalmente baixa, durante episódios de El Niño, e uma fase positiva é quando ocorre o oposto durante episódios de La Niña, e a pressão sobre a Indonésia é baixa e sobre o Pacífico ocidental é alta.[21]

Fundamentos
Diagram showing a cross-section of the Pacific and related phenomena
O Pacífico Ocidental é normalmente mais quente que o Pacífico Oriental. As águas mais quentes causam mais nebulosidade, chuvas e baixa pressão atmosférica no Pacífico Ocidental. O acúmulo de águas quentes em direção ao oeste também leva a uma camada mais espessa de água quente do oceano, o que diminui a profundidade da termoclina
Em média, a temperatura da superfície do oceano no Pacífico Oriental tropical é de aproximadamente 14-18 °C mais fria do que no Pacífico Ocidental tropical. A temperatura da superfície do mar (TSM) do Pacífico Ocidental a nordeste da Austrália é em média de 28-30 °C. As ISMs no Pacífico Leste, na costa ocidental da América do Sul, estão mais próximas de 20 °C. Os fortes ventos alísios perto do equador empurram a água para longe do Pacífico Leste e em direção ao Pacífico Oeste.[22] Esta água é lentamente aquecida pelo Sol à medida que se move para oeste ao longo do equador.[23] A superfície do oceano perto da Indonésia é tipicamente em torno de 0,46 metros mais alta do que perto do Peru devido ao acúmulo de água no Pacífico Ocidental.[24][necessário esclarecer] A termoclina, ou a zona de transição entre as águas mais quentes perto da superfície do oceano e as águas mais frias do oceano profundo,[25] é empurrada para baixo no Pacífico Ocidental devido a esta acumulação de água.[24]

O peso total de uma coluna de água do oceano é quase o mesmo no Pacífico ocidental e oriental. Como as águas mais quentes do oceano superior são ligeiramente menos densas do que as águas mais frias do oceano profundo, a camada mais espessa de água quente no Pacífico ocidental significa que a termoclina ali deve ser mais profunda. A diferença de peso deve ser suficiente para conduzir qualquer fluxo de retorno de águas profundas.[26]:12 Consequentemente, a termoclina é inclinada através do Pacífico tropical, subindo de uma profundidade média de cerca de 140 metros no Pacífico Ocidental a uma profundidade de cerca de 27 metros no Pacífico Leste.[24]

O ENOS descreve uma mudança quase periódica das condições oceânicas e atmosféricas sobre o Oceano Pacífico tropical.[22] Estas mudanças afetam os padrões climáticos em grande parte da Terra.[23] Diz-se que o Pacífico tropical está em um dos três estados do ENOS (também chamados de "fases") dependendo das condições atmosféricas e oceânicas.[27] Quando o Pacífico tropical reflete aproximadamente as condições médias, diz-se que o estado do ENOS está na fase neutra. Entretanto, o Pacífico tropical sofre mudanças ocasionais em relação a essas condições médias. Se os ventos alísios forem mais fracos que a média, o efeito da ressurgência no Pacífico Leste e o fluxo de águas superficiais oceânicas mais quentes em direção ao Pacífico Oeste diminuirão. Isso resulta em um Pacífico Ocidental mais frio e um Pacífico Leste mais quente, levando a uma mudança de nebulosidade e precipitação em direção ao Pacífico Leste. Essa situação é chamada de El Niño. O oposto ocorre se os ventos alísios forem mais fortes que a média, resultando em um Pacífico Ocidental mais quente e um Pacífico Leste mais frio. Esta situação é denominada La Niña e está associada ao aumento da nebulosidade e da precipitação no Pacífico Ocidental.[22]

ENOS descreve uma mudança quase periódica das condições oceânicas e atmosféricas sobre o Oceano Pacífico tropical.[22] Estas mudanças afetam os padrões climáticos em grande parte da Terra.[23] Diz-se que o Pacífico tropical está em um dos três estados do ENOS (também chamados de "fases") dependendo das condições atmosféricas e oceânicas.[28] Quando o Pacífico tropical reflete aproximadamente as condições médias, diz-se que o estado do ENOS é neutro. Entretanto, o Pacífico tropical sofre mudanças ocasionais em relação a essas condições médias. Se os ventos alísios forem mais fracos que a média, o efeito da ressurgência no Pacífico Leste e o fluxo de águas superficiais oceânicas mais quentes em direção ao Pacífico Oeste diminuirão. Isso resulta em um Pacífico Ocidental mais frio e um Pacífico Leste mais quente, levando a uma mudança de nebulosidade e precipitação em direção ao Pacífico Leste. Essa situação é chamada de El Niño. O oposto ocorre se os ventos alísios forem mais fortes que a média, resultando em um Pacífico Ocidental mais quente e um Pacífico Leste mais frio. Esta situação é denominada La Niña e está associada ao aumento da nebulosidade e da precipitação no Pacífico Ocidental.[22]

Feedback de Bjerknes
A estreita relação entre as temperaturas dos oceanos e a força dos ventos alísios foi identificada pela primeira vez por Jacob Bjerknes em 1969. Bjerknes também levantou a hipótese de que o ENOS era um sistema de feedback positivo onde as mudanças associadas em um componente do sistema climático (o oceano ou a atmosfera) tendem a reforçar as mudanças no outro.[29]:86 Por exemplo, durante o El Niño, o contraste reduzido nas temperaturas do oceano no Pacífico resulta em ventos alísios mais fracos, reforçando ainda mais seu estado. Este processo é conhecido como feedback de Bjerknes.[30] Embora estas mudanças associadas no oceano e na atmosfera ocorram frequentemente em conjunto, o estado da atmosfera pode assemelhar-se a uma fase do ENOS diferente do estado do oceano ou vice-versa.[31] Como os seus estados estão intimamente ligados, as variações do ENOS podem surgir de alterações tanto no oceano como na atmosfera e não necessariamente de uma alteração inicial exclusiva de um ou de outro.[32][30] Modelos conceituais que explicam como o ENOS opera geralmente aceitam a hipótese do feedback de Bjerknes. No entanto, o ENOS permaneceria perpetuamente numa única fase se o feedback de Bjerknes fosse o único processo a ocorrer.[29]:88 Várias teorias foram propostas para explicar como o ENOS pode mudar de um estado para outro, apesar do feedback positivo.[33] Estas explicações dividem-se, em linhas gerais, em duas categorias.[34] Numa visão, o feedback de Bjerknes desencadeia naturalmente feedbacks negativos[necessário esclarecer] que acabam revertendo o estado anormal do Pacífico tropical. Esta perspectiva implica que os processos que conduzem ao El Niño e à La Niña também acabam por chegar ao seu fim, tornando o ENOS um processo autossustentável.[necessário esclarecer][29]:88 Outras teorias consideram que o estado do ENOS é alterado por fenômenos irregulares e externos, como a oscilação Madden-Julian, ondas de instabilidade tropical e rajadas de vento ocidental.[29]:90

Circulação de Walker

Fase neutra: Os ventos equatoriais reúnem piscinas de água quente em direção ao oeste. A acumulação de água quente no oeste gera convecção atmosférica profunda. No leste, os ventos locais fazem com que água fria rica em nutrientes suba no Equador e ao longo da costa sul-americana.

Fase El Niño: A piscina de água quente se aproxima da costa sul-americana. A ausência de ressurgência fria aumenta o aquecimento. A água quente e a convecção atmosférica movem-se para leste. Em El Niños fortes, a termoclina mais profunda ao largo da América do Sul significa que a água que ressurge é quente e pobre em nutrientes.

Fase La Niña: A água quente está mais a oeste do que o normal.
As três fases do ENOS estão relacionadas à circulação de Walker, que recebeu esse nome em homenagem a Gilbert Walker, que descobriu a Oscilação Sul no início do século XX. A circulação de Walker é uma circulação de rotação leste-oeste nas proximidades do equador no Pacífico. O ar ascendente está associado a altas temperaturas do mar, convecção e precipitação, enquanto o ramo descendente ocorre sobre temperaturas mais frias da superfície do mar no leste. Durante o El Niño, à medida que as temperaturas da superfície do mar mudam, a Circulação de Walker também muda. O aquecimento no Pacífico tropical oriental enfraquece ou reverte o ramo descendente, enquanto as condições mais frias no oeste levam a menos chuva e ar descendente, de modo que a Circulação de Walker enfraquece primeiro e pode reverter.[35]:185

Oscilação Sul

Fase neutra: Os ventos equatoriais reúnem piscinas de água quente em direção ao oeste. A acumulação de água quente no oeste gera convecção atmosférica profunda. No leste, os ventos locais fazem com que água fria rica em nutrientes suba no Equador e ao longo da costa sul-americana.

Fase El Niño: A piscina de água quente se aproxima da costa sul-americana. A ausência de ressurgência fria aumenta o aquecimento. A água quente e a convecção atmosférica movem-se para leste. Em El Niños fortes, a termoclina mais profunda ao largo da América do Sul significa que a água que ressurge é quente e pobre em nutrientes.

Fase La Niña: A água quente está mais a oeste do que o normal.
A Oscilação do Sul é o componente atmosférico do ENOS. Este componente é uma oscilação na pressão atmosférica da superfície entre as águas tropicais do leste e do oeste do Oceano Pacífico. A força da Oscilação do Sul é medida pelo Índice de Oscilação Sul (IOS). O IOS é calculado a partir das flutuações na diferença de pressão atmosférica na superfície entre o Taiti (no Pacífico) e Darwin, na Austrália (no Oceano Índico).[36]

Embora o Índice de Oscilação do Sul tenha um longo registro de dados que remonta a 1800, sua confiabilidade é limitada devido às latitudes de Darwin e do Taiti estarem bem ao sul do Equador, de modo que a pressão atmosférica da superfície em ambos os locais está menos diretamente relacionada ao ENOS.[37] Para superar este efeito, foi criado um novo índice, denominado Índice de Oscilação Equatorial do Sul (IOES).[37][38] Para gerar esse índice, foram definidas duas novas regiões, centradas no Equador. A região ocidental está localizada sobre a Indonésia e a oriental sobre o Pacífico equatorial, próximo à costa sul-americana.[37] No entanto, os dados sobre o IOES remontam apenas a 1949.[37]

A topografia da superfície do mar muda para cima ou para baixo em vários centímetros na região equatorial do Pacífico: El Niño causa uma anomalia positiva (elevação do nível do mar) devido à expansão térmica, enquanto La Niña causa uma anomalia negativa (redução do nível do mar) por meio da contração.[39]

Três fases da temperatura da superfície do mar
Ver também: Temperatura do oceano e Temperatura da superfície do mar
O El Niño-Oscilação do Sul é um fenômeno climático único que flutua quase periodicamente entre três fases: Neutra, La Niña ou El Niño.[12] La Niña e El Niño são fases opostas que exigem que certas mudanças ocorram tanto no oceano quanto na atmosfera antes que um evento seja declarado.[12] A fase fria do ENOS é La Niña, com o SST no Pacífico oriental abaixo da média e pressão atmosférica alta no Pacífico oriental e baixa no Pacífico ocidental. O ciclo ENOS, incluindo El Niño e La Niña, causa mudanças globais na temperatura e na precipitação.[40][41]


Fase neutra: Os ventos equatoriais reúnem piscinas de água quente em direção ao oeste. A acumulação de água quente no oeste gera convecção atmosférica profunda. No leste, os ventos locais fazem com que água fria rica em nutrientes suba no Equador e ao longo da costa sul-americana.

Fase El Niño: A piscina de água quente se aproxima da costa sul-americana. A ausência de ressurgência fria aumenta o aquecimento. A água quente e a convecção atmosférica movem-se para leste. Em El Niños fortes, a termoclina mais profunda ao largo da América do Sul significa que a água que ressurge é quente e pobre em nutrientes.

Fase La Niña: A água quente está mais a oeste do que o normal.
Fase neutra
Se a variação de temperatura da climatologia estiver dentro de 0,5 °C, as condições ENOS são descritas como neutras. Condições neutras são a transição entre as fases quente e fria do ENOS. As temperaturas da superfície do mar (por definição), a precipitação tropical e os padrões de vento estão próximos das condições médias durante esta fase.[42] Quase metade de todos os anos ocorrem em períodos neutros.[43] Durante a fase ENOS neutra, outras anomalias/padrões climáticos, como o sinal da Oscilação do Atlântico Norte ou o padrão de teleconexão Pacífico-América do Norte, exercem maior influência.[44]

Fase El Niño

Ciclo do evento El Niño de 1997–98 mostrando anomalias extremas na temperatura da superfície do mar (TSM) no Pacífico tropical oriental
As condições de El Niño são estabelecidas quando a circulação de Walker enfraquece ou reverte e a circulação de Hadley se fortalece[45][46][necessário esclarecer] levando ao desenvolvimento de uma faixa de água oceânica quente no Pacífico equatorial central e centro-leste (aproximadamente entre a Linha Internacional de Data e 120°W), incluindo a área ao largo da costa oeste da América do Sul,[47][48] à medida que a ressurgência de água fria ocorre menos ou não ocorre em alto mar.[3]

Este aquecimento provoca uma alteração na circulação atmosférica, levando a uma maior pressão atmosférica no Pacífico ocidental e a uma menor pressão atmosférica no Pacífico oriental,[49] com a precipitação a diminuir na Indonésia, Índia e norte da Austrália, enquanto a precipitação e a formação de ciclones tropicais aumentam no Oceano Pacífico tropical.[50] Os ventos alísios de superfície de baixo nível, que normalmente sopram de leste a oeste ao longo do equador, enfraquecem ou começam a soprar na outra direção.[44]

Sabe-se que as fases do El Niño ocorrem em intervalos irregulares de dois a sete anos e duram de nove meses a dois anos.[51] A duração média do período é de cinco anos. Quando este aquecimento ocorre durante sete a nove meses, é classificado como “condições” de El Niño; quando a sua duração é mais longa, é classificado como um “episódio” de El Niño.[52]


Anomalias na temperatura da superfície do mar em novembro 2007, mostrando condições do La Niña
Acredita-se que tenham ocorrido pelo menos 30 eventos El Niño entre 1900 e 2024, com os eventos de 1982-83, 1997-98 e 2014-16 entre os mais fortes já registados.[53] Desde 2000, eventos El Niño foram observados em 2002–03, 2004–05, 2006–07, 2009–10, 2014–16, 2018–19,[54][55][56] e 2023–24.[57][58]

Os principais eventos ENOS foram registrados nos anos de 1790–93, 1828, 1876–78, 1891, 1925–26, 1972–73, 1982–83, 1997–98, 2014–16 e 2023–24.[59][60][61] Durante episódios fortes do El Niño, um pico secundário na temperatura da superfície do mar no extremo leste do Oceano Pacífico equatorial às vezes segue o pico inicial.[62]

Uma circulação de Walker especialmente forte causa o La Niña, que é considerado a fase oceânica fria e atmosférica positiva do fenômeno climático El Niño-Oscilação do Sul (ENOS), bem como o oposto do padrão climático El Niño,[19] onde a temperatura da superfície do mar na parte equatorial oriental do Oceano Pacífico central será menor do que o normal em 3–5 °C. O fenômeno ocorre quando ventos fortes sopram água quente da superfície do oceano para longe da América do Sul, através do Oceano Pacífico, em direção à Indonésia.[19] À medida que essa água quente se move para o oeste, a água fria do mar profundo sobe à superfície perto da América do Sul.[19]

O movimento de tanto calor por um quarto do planeta, e particularmente na forma de temperatura na superfície do oceano, pode ter um efeito significativo no clima de todo o planeta. Ondas de instabilidade tropical visíveis nos mapas de temperatura da superfície do mar, mostrando uma nuvem de água mais fria, estão frequentemente presentes durante períodos neutros ou em condições de La  Niña.[63]

Fases de transição
Fases de transição no início ou no fim do El Niño ou La Niña também podem ser fatores importantes no clima global ao afetar as teleconexões. Os episódios mais significativos, conhecidos como Trans-Niño, são medidos pelo índice Trans-Niño (ITN).[64] Exemplos de clima afetado por uma curta duração na América do Norte incluem precipitação no noroeste dos EUA[65] e intensa atividade de tornados nos EUA contíguos.[66]

Variações
ENOS Modoki
O primeiro padrão ENOS a ser reconhecido, denominado ENOS do Pacífico Oriental (EP), para distingui-lo de outros,[67] envolve anomalias de temperatura no Pacífico oriental. Entretanto, nas décadas de 1990 e 2000, foram observadas variações nas condições do ENOS, nas quais o local habitual da anomalia de temperatura (Niño 1 e 2) não é afetado, mas também surge uma anomalia no Pacífico central (Niño 3 e 4).[68] O fenômeno é chamado de ENOS do Pacífico Central (CP),[67] ENOS "linha de data" (porque a anomalia surge perto da linha de data), ou ENOS "Modoki" (Modoki em japonês significa "semelhante, mas diferente").[69][70] Existem variações de ENOS adicionais aos tipos EP e CP, e alguns cientistas argumentam que o ENOS existe como um continuum, frequentemente com tipos híbridos.[71]


Mapa mostrando Niño/Niña nas regiões 1 a 4, 3 e 4 sendo oeste e extremo oeste e muito maiores que 1 e 2, uma zona costeira peruana/equatoriana que difere sutilmente de norte a sul
Os efeitos do CP ENOS são diferentes daqueles do EP ENOS. O El Niño Modoki está associado a um número cada vez maior de furacões que atingem o Atlântico com frequência crescente.[72] O La Niña Modoki leva a um aumento da precipitação no noroeste da Austrália e no norte da bacia Murray-Darling, em vez de na parte oriental do país, como num EP La Niña convencional.[73] Além disso, La Niña Modoki aumenta a frequência de tempestades ciclônicas na Baía de Bengala, mas diminui a ocorrência de tempestades severas no Oceano Índico em geral.[74]

O primeiro El Niño registado que teve origem no Pacífico central e se deslocou para leste foi em 1986.[75] Os recentes El Niños do Pacífico Central ocorreram em 1986–87, 1991–92, 1994–95, 2002–03, 2004–05 e 2009–10.[76] Além disso, houve eventos "Modoki" em 1957-59,[77] 1963-64, 1965-66, 1968-70, 1977-78 e 1979-80.[78][79] Algumas fontes dizem que os El Niños de 2006-07 e 2014-16 também foram El Niños do Pacífico Central.[80][81] Os eventos La Niña Modoki mais recentes ocorreram em 1973–1974, 1975–1976, 1983–1984, 1988–1989, 1998–1999, 2000–2001, 2008–2009, 2010–2011 e 2016–2017.[82][83]

A recente descoberta do ENOS Modoki levou alguns cientistas a acreditar que ele está relacionado ao aquecimento global.[84] No entanto, dados abrangentes de satélite remontam apenas a 1979. Mais pesquisas precisarão ser feitas para encontrar a correlação e estudar episódios anteriores do El Niño. De um modo mais geral, não existe consenso científico sobre como/se as alterações climáticas poderão afetar o ENOS.[11]

Há também um debate científico sobre a própria existência deste "novo" ENOS. Vários estudos contestam a realidade desta distinção estatística ou a sua ocorrência crescente, ou ambas, argumentando que o registo fiável é demasiado curto para detectar tal distinção,[85][86] não encontrando qualquer distinção ou tendência utilizando outras abordagens estatísticas,[87][88][89][90][91] ou que outros tipos devem ser distinguidos, como o ENOS padrão e o ENOS extremo.[92][93]

Da mesma forma, seguindo a natureza assimétrica das fases quentes e frias do ENOS, alguns estudos não conseguiram identificar variações semelhantes para La Niña, tanto nas observações como nos modelos climáticos,[94] mas algumas fontes conseguiram identificar variações em La Niña com águas mais frias no Pacífico central e temperaturas médias ou mais quentes da água tanto no Pacífico oriental como ocidental, mostrando também correntes do Oceano Pacífico oriental indo na direção oposta em comparação com as correntes em La Niñas tradicionais.[69][70][95]

ENOS Costero
Cunhado pelo Comité Multissetorial Peruano Encargado del Estudio Nacional del Fenómeno El Niño (ENFEN),[96] ENOS Costero, ou ENOS Oriental, é o nome dado ao fenômeno onde as anomalias de temperatura da superfície do mar se concentram principalmente na costa sul-americana, especialmente do Peru e do Equador.[97] Estudos apontam diversos fatores que podem levar à sua ocorrência,[98] às vezes acompanhando, ou sendo acompanhado, por uma maior ocorrência de EP ENOS, ou mesmo apresentando condições opostas às observadas nas demais regiões de Niño quando acompanhadas por variações de Modoki.[99]

Os eventos do ENOS Costero geralmente apresentam efeitos mais localizados, com fases quentes que levam ao aumento das chuvas na costa do Equador, norte do Peru e floresta amazônica, e aumento das temperaturas na costa norte do Chile,[96][100] e fases frias que levam à seca na costa peruana, e aumento das chuvas e diminuição das temperaturas em suas regiões montanhosas e de selva.[101]

Como não influenciam o clima global tanto quanto os outros tipos, esses eventos apresentam correlações menores e mais fracas com outras características significativas do ENOS, não sendo sempre desencadeados por ondas Kelvin,[96] nem acompanhados por respostas proporcionais da Oscilação do Sul.[102] De acordo com o Coastal Niño Index (ICEN), os eventos fortes de El Niño Costero incluem os anos de 1957, 1982–83, 1997–98 e 2015–16, e os eventos de La Niña Costera incluem 1950, 1954–56, 1962, 1964, 1966, 1967–68, 1970–71, 1975–76 e 2 013.[103]

Monitoramento e declaração de condições

As várias "regiões Niño" onde as temperaturas da superfície do mar são monitoradas para determinar a fase ENSO atual (quente ou fria)
Atualmente, cada país tem um limiar diferente para o que constitui um evento El Niño, que é adaptado aos seus interesses específicos, por exemplo:[5]

Nos Estados Unidos, o seu Centro de Previsão Climática e o Instituto Internacional de Pesquisa sobre Clima e Sociedade monitorizam as temperaturas da superfície do mar na região Niño 3.4, a atmosfera tropical do Pacífico e prevêem que o Índice Niño Oceânico da NOAA será igual ou superior a .5 °C por várias temporadas consecutivas.[104] A região Niño 3.4 se estende dos meridianos 120 a 170 de longitude oeste, a cinco graus de latitude de cada lado do equador, sendo constantemente monitorada. São aproximadamente 3 mil quilômetros ao sudeste do Havaí. A média dos últimos três meses para a área é calculada e, se a temperatura da região for maior que 0,5 °C acima (ou abaixo) do normal para esse período, então um El Niño (ou La Niña) é considerado em andamento.[105]
O Australian Bureau of Meteorology analisa os ventos alísios, o Índice de Oscilação do Sul, os modelos meteorológicos e as temperaturas da superfície do mar nas regiões Niño 3 e 3.4, antes de declarar um evento ENOS.[106]
A Agência Meteorológica do Japão declara que um evento ENOS começou quando o desvio médio da temperatura da superfície do mar durante cinco meses para a região Niño 3 for superior a 0.5 °C por seis meses consecutivos ou mais.[107]
O governo peruano declara que um ENOS Costero está em andamento se o desvio da temperatura da superfície do mar nas regiões Niño 1+2 for igual ou superior 0.4 °C por pelo menos três meses.[103]
O Met Office do Reino Unido também usa um período de vários meses para determinar o estado ENOS.[108] Quando este aquecimento ou arrefecimento ocorre apenas durante sete a nove meses, é classificado como “condições” de El Niño/La Niña; quando ocorre durante mais do que esse período, é classificado como “episódios” de El Niño/La Niña.[109]
Efeitos do ENOS no clima global
Refer to caption
Esta imagem mostra três exemplos de variabilidade climática interna medidos entre 1950 e 2012: a oscilação El Niño-Sul, a oscilação Ártica e a oscilação do Atlântico Norte.
Na ciência das alterações climáticas, o ENOS é conhecido como um dos fenômenos de variabilidade climática interna.[necessário esclarecer] Os outros dois principais[necessário esclarecer] são a oscilação decadal do Pacífico e a oscilação multidecadal do Atlântico.[10]:23

La Niña impacta o clima global e perturba os padrões climáticos normais, o que pode levar a tempestades intensas em alguns lugares e secas em outros.[110] Os eventos El Niño causam picos de curto prazo (aproximadamente 1 ano de duração) na temperatura média global da superfície, enquanto os eventos La Niña causam resfriamento de curto prazo.[8] Portanto, a frequência relativa dos eventos El Niño em comparação com os eventos La Niña pode afetar as tendências da temperatura global em escalas de tempo decadais.[9]

Mudanças climáticas
Não há sinais de que haja mudanças reais no fenômeno físico ENOS devido às mudanças climáticas. Os modelos climáticos não simulam o ENOS bem o suficiente para fazer previsões confiáveis. As tendências futuras deste fenômeno são incertas[11] uma vez que diferentes modelos fazem previsões diferentes.[111][112] Pode ser que o fenômeno observado de eventos El Niño mais frequentes e mais fortes ocorra apenas na fase inicial do aquecimento global e, então (por exemplo, depois que as camadas inferiores do oceano também esquentam), o El Niño se tornará mais fraco.[113] Também pode ser que as forças estabilizadoras e desestabilizadoras que influenciam o fenômeno[necessário esclarecer] acabarão por compensar-se mutuamente.[114]

As consequências do ENOS em termos de anomalias de temperatura, precipitação e eventos climáticos extremos em todo o mundo estão claramente aumentando e estão associadas às mudanças climáticas. Por exemplo, estudos recentes (desde cerca de 2019) descobriram que as alterações climáticas estão a aumentar a frequência de eventos extremos do El Niño.[115][116] Anteriormente não havia consenso sobre se as alterações climáticas teriam alguma influência na intensidade ou duração dos eventos El Niño, uma vez que a investigação apoiava alternadamente que os eventos El Niño se tornariam mais fortes e mais fracos, mais longos e mais curtos.[117][118]

Nas últimas décadas, o número de eventos El Niño aumentou e o número de eventos La Niña diminuiu,[119] embora seja necessária a observação do ENOS durante muito mais tempo para detectar mudanças robustas.[120]

Estudos de dados históricos mostram que a variação recente do El Niño provavelmente está ligada ao aquecimento global. Por exemplo, alguns resultados, mesmo após subtrair a influência positiva da variação decadal, mostram-se possivelmente presentes na tendência do ENOS,[121] a amplitude da variabilidade do ENOS nos dados observados ainda aumenta, em até 60% nos últimos 50 anos.[122] Um estudo publicado em 2023 por investigadores da CSIRO concluiu que as alterações climáticas podem ter aumentado em duas vezes a probabilidade de eventos El Niño fortes e nove vezes a probabilidade de eventos La Niña fortes.[123][124] O estudo afirmou ter encontrado um consenso entre diferentes modelos e experimentos.[125]

O Sexto Relatório de Avaliação do IPCC resumiu o estado da arte da pesquisa em 2021 sobre o futuro do ENOS da seguinte forma:

“A longo prazo, é muito provável que a variação da precipitação relacionada com o El Niño-Oscilação do Sul aumente”[10]:113
“É muito provável que a variabilidade da precipitação relacionada com as alterações na força e na extensão espacial das teleconexões do ENOS conduza a alterações significativas à escala regional”.[10]:114
"Há uma confiança média de que tanto a amplitude do ENOS como a frequência de eventos de alta magnitude desde 1950 são maiores do que no período de 1850 e possivelmente até 1400".[10]:373
Investigações sobre pontos de inflexão
O ENOS é considerado um potencial elemento de inflexão no clima da Terra.[126] O aquecimento global pode fortalecer a teleconexão ENOS e os eventos climáticos extremos resultantes.[127] Por exemplo, um aumento na frequência e magnitude dos eventos El Niño desencadeou temperaturas mais altas do que o normal no Oceano Índico, através da modulação da circulação de Walker.[128] Isto resultou num rápido aquecimento do Oceano Índico e, consequentemente, num enfraquecimento das monções asiáticas.[129]


Lista anterior (2008) de elementos decisivos no sistema climático.
A possibilidade de o El Niño-Oscilação do Sul ser um elemento de inflexão atraiu atenção no passado. Normalmente, ventos fortes sopram para oeste através do Oceano Pacífico Sul, da América do Sul à Austrália. A cada dois a sete anos, os ventos enfraquecem devido a mudanças de pressão e o ar e a água no meio do Pacífico aquecem, causando mudanças nos padrões de movimento do vento em todo o mundo. Isto é conhecido como El Niño e normalmente leva a secas na Índia, Indonésia e Brasil, e ao aumento de inundações no Peru. Em 2015/2016, isto causou escassez de alimentos que afetou mais de 60 milhões de pessoas.[130] As secas induzidas pelo El Niño podem aumentar a probabilidade de incêndios florestais na Amazônia.[131] O limite para um "tombamento" foi estimado entre 3,5 e 7 °C de aquecimento global em 2016. Após o tombamento, o sistema ficaria num estado de El Niño mais permanente, em vez de oscilar entre diferentes estados. Isto aconteceu no passado da Terra, no Plioceno, mas a configuração do oceano era significativamente diferente da atual. Até o momento, não há evidências definitivas que indiquem uma mudança no comportamento do ENOS,[131] e o Sexto Relatório de Avaliação do IPCC concluiu que é "virtualmente certo que o ENOS continuará a ser o modo dominante de variabilidade interanual num mundo mais quente".[132] Consequentemente, a avaliação de 2022 já não o inclui na lista de prováveis ​​elementos de ruptura.

Efeitos do ENOS nos padrões climáticos
Ciclones tropicais
A maioria dos ciclones tropicais se forma no lado da cordilheira subtropical mais próximo do equador, depois se move em direção ao pólo, passando pelo eixo da cordilheira antes de retornar ao cinturão principal dos ventos do oeste.[133] Áreas a oeste do Japão e da Coreia tendem a sofrer muito menos impactos de ciclones tropicais de setembro a novembro durante o El Niño e anos neutros. Durante os anos de El Niño, a quebra[necessário esclarecer] na cordilheira subtropical tende a ficar perto de 130°E, o que favoreceria o arquipélago japonês.[134]

Com base na energia ciclônica acumulada (ECA) modelada e observada, os anos de El Niño geralmente resultam em temporadas de furacões menos ativas no Oceano Atlântico, mas favorecem uma mudança para a atividade de ciclones tropicais no Oceano Pacífico, em comparação com os anos de La Niña que favorecem o desenvolvimento de furacões acima da média no Atlântico e menos na bacia do Pacífico.[135]

Sobre o Oceano Atlântico, o cisalhamento vertical do vento aumenta, o que inibe a gênese e a intensificação dos ciclones tropicais, ao fazer com que os ventos de oeste sejam mais fortes.[136] A atmosfera sobre o Oceano Atlântico também pode ser mais seca e estável durante eventos El Niño, o que pode inibir a gênese e a intensificação de ciclones tropicais.[136] Na bacia do Pacífico Oriental: os eventos El Niño contribuem para a diminuição do cisalhamento vertical do vento de leste e favorecem a atividade de furacões acima do normal.[137] No entanto, os impactos do estado ENOS nesta região podem variar e são fortemente influenciados pelos padrões climáticos de fundo.[137] A bacia do Pacífico Ocidental sofre uma mudança na localização onde os ciclones tropicais se formam durante os eventos El Niño, com a formação dos ciclones tropicais a deslocar-se para leste, sem uma alteração significativa na quantidade de ciclones que se desenvolvem a cada ano.[136] Como resultado desta mudança, a Micronésia tem maior probabilidade, e a China menos probabilidade, de ser afetada por ciclones tropicais.[134] Uma mudança na localização onde os ciclones tropicais se formam também ocorre no Oceano Pacífico Sul entre 135°E e 120°W, sendo mais provável que os ciclones tropicais ocorram na bacia do Pacífico Sul do que na região australiana.[138][136] Como resultado desta mudança, os ciclones tropicais têm 50% menos probabilidades de atingir Queensland, enquanto que o risco de um ciclone tropical é elevado para nações insulares como Niue, Polinésia Francesa, Tonga, Tuvalu e as Ilhas Cook.[138][139][140]

Influência remota no Oceano Atlântico tropical
Um estudo de registros climáticos mostrou que os eventos El Niño no Pacífico equatorial estão geralmente associados a um Atlântico Norte tropical quente na primavera e no verão seguintes.[141] Cerca de metade dos eventos El Niño persistem o suficiente nos meses de primavera para que o Poço Quente do Hemisfério Ocidental se torne invulgarmente grande no verão.[142] Ocasionalmente, o efeito do El Niño na circulação do Atlântico Walker sobre a América do Sul fortalece os ventos alísios de leste na região do Atlântico equatorial ocidental. Como resultado, pode ocorrer um resfriamento incomum no Atlântico equatorial oriental na primavera e no verão, após os picos do El Niño no inverno.[143] Casos de eventos do tipo El Niño em ambos os oceanos simultaneamente foram associados a fomes severas relacionadas com a ausência prolongada de chuvas de monções.[144]
"""
    summary_template = """
    dada a informação {information} eu quero que você crie:
    1. um resumo pequeno
    2. dois fatos interessantes sobre ela
"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    ) #Um objeto que separa a instrução fixa(summary template) das variáveis dinâmicas {information}

    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0
    )

    # llm = ChatOllama(model="gemma3:270m", temperature=0)

    #   a chain diz pegue a saída a esquerda(PromptTemplate), que gera PromptValue, e 
    # passe como entrada para o elemento da direita ChatOpenAi
    # beneficio de utilizar | é que ganha suporte nativo a execução assíncrona, paralelismo em lote,
    # streaming de tokens e rastreamento
    chain = summary_prompt_template | llm 

    #chain.invoke método padrão da interface Runnable que dispara a execução síncrona passando o dicionário de entrada
    response = chain.invoke(input={"information": information})
    print(response.content)
if __name__ == "__main__":
    main()