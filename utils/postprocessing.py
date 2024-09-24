def clean_answer(answer):
    # Bersihkan jawaban dari karakter yang tidak diperlukan atau hal-hal lain yang tidak diinginkan
    cleaned_answer = answer.strip().replace("\n", " ").replace("\t", " ")
    return cleaned_answer

def format_output(answer, reference, translation=None, summary=None):
    # Format output untuk ditampilkan ke user
    output = f"Jawaban: {clean_answer(answer)}\n"
    output += f"Referensi: {reference}\n"
    
    if translation:
        output += f"Terjemahan: {translation}\n"
    
    if summary:
        output += f"Ringkasan: {summary}\n"
    
    return output
