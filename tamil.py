import os
import json

# The dictionary
label_to_tamil = {
    'a': 'அ', 'aa': 'ஆ', 'ba': 'ப', 'baa': 'பா', 'bai': 'பை', 'bau': 'பௌ',
    'be': 'பே', 'bee': 'பீ', 'bha': 'ப்ஹ', 'bhaa': 'ப்ஹா', 'bhai': 'ப்ஹை',
    'bhau': 'ப்ஹௌ', 'bhee': 'ப்ஹீ', 'bhi': 'ப்ஹி', 'bho': 'ப்ஹோ', 'bhoo': 'ப்ஹூ',
    'bhu': 'ப்ஹு', 'bhya': 'ப்ஹ்யா', 'bi': 'பி', 'bo': 'போ', 'boo': 'பூ', 'bra': 'ப்ரா',
    'bu': 'பு', 'cha': 'ச', 'chaa': 'சா', 'chai': 'சை', 'chau': 'சௌ', 'che': 'சே',
    'chee': 'சீ',
    'chha': 'ச்ஹ',
    'chhaa': 'ச்ஹா',
    'chhai': 'ச்ஹை',
    'chhau': 'ச்ஹௌ',
    'chhe': 'ச்ஹே',
    'chhee': 'ச்ஹீ',
    'chhi': 'ச்ஹி',
    'chho': 'ச்ஹோ',
    'chhoo': 'ச்ஹூ',
    'chhu': 'ச்ஹு',
    'chi': 'சி',
    'cho': 'சோ',
    'choo': 'சூ',
    'chu': 'சு',
    'cya': 'ச்யா',
    'da': 'த',
    'dda': 'த்த',
    'daa': 'தா',
    'ddaa': 'த்தா',
    'dai': 'தை',
    'ddai': 'த்தை',
    'dau': 'தௌ',
    'ddau': 'த்தௌ',
    'de': 'தே',
    'dde': 'த்தே',
    'dee': 'தீ',
    'ddee': 'த்தீ',
    'dha': 'த்ஹ',
    'ddha': 'த்த்ஹ',
    'dhaa': 'த்ஹா',
    'ddhaa': 'த்த்ஹா',
    'dhai': 'த்ஹை',
    'ddhai': 'த்த்ஹை',
    'dhau': 'த்ஹௌ',
    'ddhau': 'த்த்ஹௌ',
    'dhe': 'த்ஹே',
    'ddhe': 'த்த்ஹே',
    'dhee': 'த்ஹீ',
    'ddhee': 'த்த்ஹீ',
    'dhi': 'த்ஹி',
    'ddhi': 'த்த்ஹி',
    'dho': 'த்ஹோ',
    'ddho': 'த்த்ஹோ',
    'dhoo': 'த்ஹூ',
    'ddhoo': 'த்த்ஹூ',
    'dhu': 'த்ஹு',
    'ddhu': 'த்த்ஹு',
    'dhya': 'த்ஹ்யா',
    'di': 'தி',
    'ddi': 'த்தி',
    'do': 'தோ',
    'ddo': 'த்தோ',
    'doo': 'தூ',
    'ddoo': 'த்தூ',
    'du': 'து',
    'ddu': 'த்து',
    'dva': 'த்வா',
    'e': 'எ',
    'ga': 'க',
    'gaa': 'கா',
    'gai': 'கை',
    'gau': 'கௌ',
    'ge': 'கே',
    'gee': 'கீ',
    'gha': '஘',
    'ghaa': '஘ா',
    'ghai': '஘ை',
    'ghau': '஘ௌ',
    'ghe': '஘ே',
    'ghee': '஘ீ',
    'ghi': '஘ி',
    'gho': '஘ோ',
    'ghoo': '஘ூ',
    'ghu': '஘ு',
    'gi': 'கி',
    'go': 'கோ',
    'goo': 'கூ',
    'gu': 'கு',
    'gya': 'ஞா',
    'ha': 'ஹ',
    'haa': 'ஹா',
    'hai': 'ஹை',
    'hau': 'ஹௌ',
    'he': 'ஹே',
    'hee': 'ஹீ',
    'hi': 'ஹி',
    'hima': 'ஹிமா',
    'hma': 'ஹ்மா',
    'ho': 'ஹோ',
    'hoo': 'ஹூ',
    'hu': 'ஹு',
    'hva': 'ஹ்வா',
    'hya': 'ஹ்யா',
    'i': 'இ',
    'ja': 'ஜ',
    'jaa': 'ஜா',
    'jai': 'ஜை',
    'jau': 'ஜௌ',
    'je': 'ஜே',
    'jee': 'ஜீ',
    'Jha': 'ஜ்ஹ',
    'jhaa': 'ஜ்ஹா',
    'jhai': 'ஜ்ஹை',
    'jhau': 'ஜ்ஹௌ',
    'jhe': 'ஜ்ஹே',
    'jhee': 'ஜ்ஹீ',
    'jhi': 'ஜ்ஹி',
    'jho': 'ஜ்ஹோ',
    'jhoo': 'ஜ்ஹூ',
    'jhu': 'ஜ்ஹு',
    'ji': 'ஜி',
    'jo': 'ஜோ',
    'joo': 'ஜூ',
    'ju': 'ஜு',
    'ka': 'க',
    'kaa': 'கா',
    'kai': 'கை',
    'kau': 'கௌ',
    'ke': 'கே',
    'kee': 'கீ',
    'kha': 'க்ஹ',
    'khaa': 'க்ஹா',
    'khai': 'க்ஹை',
    'khau': 'க்ஹௌ',
    'khe': 'க்ஹே',
    'khee': 'க்ஹீ',
    'khi': 'க்ஹி',
    'kho': 'க்ஹோ',
    'khoo': 'க்ஹூ',
    'khu': 'க்ஹு',
    'khya': 'க்ஹ்யா',
    'ki': 'கி',
    'ko': 'கோ',
    'koo': 'கூ',
    'kra': 'கிரா',
    'ku': 'கு',
    'kya': 'க்யா',
    'la': 'ல',
    'laa': 'லா',
    'lai': 'லை',
    'lau': 'லௌ',
    'le': 'லே',
    'lee': 'லீ',
    'li': 'லி',
    'lo': 'லோ',
    'loo': 'லூ',
    'lu': 'லு',
    'ma': 'ம',
    'maa': 'மா',
    'mai': 'மை',
    'mau': 'மௌ',
    'me': 'மெ',
    'mee': 'மீ',
    'mha': 'ம்ஹ',
    'mi': 'மி',
    'mo': 'மோ',
    'moo': 'மூ',
    'mu': 'மு',
    'mya': 'ம்யா',
    'na': 'ந',
    'nna': 'ந்ந',
    'naa': 'நா',
    'nnaa': 'ந்நா',
    'nai': 'நை',
    'nnai': 'ந்நை',
    'nau': 'நௌ',
    'nnau': 'ந்நௌ',
    'ne': 'நே',
    'nne': 'ந்நே',
    'ṇee': 'நீ',
    'nnee': 'ந்நீ',
    'ni': 'நி',
    'ṇi': 'ந்நி',
    'no': 'நோ',
    'nno': 'ந்நோ',
    'noo': 'நூ',
    'nnoo': 'ந்நூ',
    'nu': 'நு',
    'nnu': 'ந்நு',
    'nya': 'ந்யா',
    'o': 'ஓ',
    'pa': 'ப',
    'paa': 'பா',
    'pai': 'பை',
    'pau': 'பௌ',
    'pe': 'பே',
    'pee': 'பீ',
    'pha': 'ப்ஹ',
    'phaa': 'ப்ஹா',
    'phai': 'ப்ஹை',
    'phau': 'ப்ஹௌ',
    'phe': 'ப்ஹே',
    'phee': 'ப்ஹீ',
    'phi': 'ப்ஹி',
    'pho': 'ப்ஹோ',
    'phoo': 'ப்ஹூ',
    'phu': 'ப்ஹு',
    'pi': 'பி',
    'po': 'போ',
    'poo': 'பூ',
    'pra': 'ப்ரா',
    'pta': 'ப்டா',
    'pu': 'பு',
    'ra': 'ர',
    'raa': 'ரா',
    'rai': 'ரை',
    'rau': 'ரௌ',
    'rda': 'ர்தா',
    're': 'ரே',
    'ree': 'ரீ',
    'ri': 'ரி',
    'ro': 'ரோ',
    'roo': 'ரூ',
    'rsa': 'ஆ',
    'ru': 'ரு',
    'rda': 'ர்தா',
    'sa': 'ச',
    'ssa': 'ச்ச',
    'saa': 'சா',
    'ssaa': 'ச்சா',
    'sai': 'சை',
    'ssai': 'ச்சை',
    'sau': 'சௌ',
    'se': 'சே',
    'sse': 'ச்சே',
    'see': 'சீ',
    'ssee': 'ச்சீ',
    'sha': 'ஷ',
    'shaa': 'ஷா',
    'shai': 'ஷை',
    'shau': 'ஷௌ',
    'sshau': 'ஷ்ஹௌ',
    'she': 'ஷே',
    'shee': 'ஷீ',
    'sho': 'ஷோ',
    'shoo': 'ஷூ',
    'si': 'ஷி',
    'ssi': 'ஷ்ச',
    'skha': 'ஸ்கா',
    'sma': 'ஸ்மா',
    'so': 'சோ',
    'sso': 'ஸ்சோ',
    'soo': 'சூ',
    'ssoo': 'ஸ்சூ',
    'spa': 'ஸ்பா',
    'sta': 'ஸ்டா',
    'su': 'ஸு',
    'ssu': 'ஸ்சு',
    'sva': 'ஸ்வா',
    'ssva': 'ஸ்ச்வா',
    'sya': 'ஸ்யா',
    'ta': 'த',
    'tta': 'த்த',
    'taa': 'தா',
    'ttaa': 'த்தா',
    'tai': 'தை',
    'ttai': 'த்தை',
    'tau': 'தௌ',
    'ttau': 'த்தௌ',
    'te': 'தே',
    'tte': 'த்தே',
    'tee': 'தீ',
    'ttee': 'த்தீ',
    'tha': 'த',
    'ttha': 'த்த',
    'thaa': 'தா',
    'tthaa': 'த்தா',
    'thai': 'தை',
    'ttai': 'த்தை',
    'thau': 'தௌ',
    'tthau': 'த்தௌ',
    'the': 'தே',
    'tthe': 'த்தே',
    'thee': 'தீ',
    'tthee': 'த்தீ',
    'thi': 'தி',
    'tthi': 'த்தி',
    'tho': 'தோ',
    'ttho': 'த்தோ',
    'thoo': 'தூ',
    'tthoo': 'த்தூ',
    'thu': 'து',
    'tthu': 'த்து',
    'tai': 'தை',
    'ttai': 'த்தை',
    'to': 'தோ',
    'tto': 'த்தோ',
    'too': 'தூ',
    'ttoo': 'த்தூ',
    'tra': 'த்ரா',
    'tu': 'து',
    'ttu': 'த்து',
    'tva': 'த்வா',
    'tya': 'த்யா',
    'va': 'வ',
    'vaa': 'வா',
    'vai': 'வை',
    'vau': 'வௌ',
    've': 'வே',
    'vee': 'வீ',
    'vi': 'வி',
    'vo': 'வோ',
    'voo': 'வூ',
    'vu': 'வு',
    'vya': 'வ்யா',
    'ya': 'ய',
    'yaa': 'யா',
    'yai': 'யை',
    'yau': 'யௌ',
    'ye': 'யே',
    'yi': 'யி',
    'yo': 'யோ',
    'yoo': 'யூ',
    'yu': 'யு',
    'yva': 'ய்வா'
}

# Save to a JSON file
file_name = 'label_to_tamil.json'
with open(file_name, 'w', encoding='utf-8') as json_file:
    json.dump(label_to_tamil, json_file, ensure_ascii=False, indent=4)

# Get the absolute path
absolute_path = os.path.abspath(file_name)
print("The absolute path of the JSON file is:", absolute_path)
