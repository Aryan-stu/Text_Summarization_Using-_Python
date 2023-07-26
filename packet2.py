import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

from transformers import T5Tokenizer, T5ForConditionalGeneration

def generate_summary_t5(text, num_sentences=3):

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=num_sentences*40, min_length=num_sentences*10, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def summarize_text():

    input_text = input_text_area.get("1.0", "end-1c")
    num_sentences = int(num_sentences_entry.get())

    if not input_text.strip():
        messagebox.showwarning("Input Error", "Please enter some text to summarize.")
    else:
        summary = generate_summary_t5(input_text, num_sentences=num_sentences)

        output_text_area.config(state="normal")
        output_text_area.delete("1.0", "end")
        output_text_area.insert("1.0", summary)
        output_text_area.config(state="disabled")

root = tk.Tk()
root.title("Text Summarization Tool")

input_label = tk.Label(root, text="Enter Text to Summarize:")
input_label.pack()
input_text_area = ScrolledText(root, height=8, wrap=tk.WORD)
input_text_area.pack()

num_sentences_label = tk.Label(root, text="Number of Sentences in the Summary:")
num_sentences_label.pack()
num_sentences_entry = tk.Entry(root)
num_sentences_entry.insert(0, "3")
num_sentences_entry.pack()

summarize_button = tk.Button(root, text="Generate Summary", command=summarize_text)
summarize_button.pack()

# Output text area
output_label = tk.Label(root, text="Summary:")
output_label.pack()
output_text_area = ScrolledText(root, height=6, wrap=tk.WORD, state="disabled")
output_text_area.pack()

# Start the main event loop
root.mainloop()
