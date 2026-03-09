""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols: Todo: should write more elegently!!
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(1024)]  # 1024 by daxin
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(2048)]  # 2048 by dehua
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(4096)]  # 4096 by daxin

symbols_with_1024 = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(1024)]  # 1024 by daxin
symbols_with_2048 = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(2048)]  # 2048 by dehua
symbols_with_4096 = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + [str(v) for v in range(4096)]  # 4096 by daxin

# Special symbol ids
SPACE_ID = symbols.index(" ")
