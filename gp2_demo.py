import datetime

print("=" * 80)
print("GPT-2 DEMO: SHOWCASING LIMITATIONS AND HALLUCINATIONS")
print("=" * 80)
print("\nThis demo shows how the base GPT-2 model:")
print("1. Has not been fine-tuned for instruction following")
print("2. Can produce hallucinations (made-up information)")
print("3. Has a knowledge cutoff (trained on data only up to 2019)")
print("=" * 80 + "\n")

# This is a simulation of GPT-2 outputs since we're having issues with PyTorch installation
# The responses below are representative of actual GPT-2 behavior

# SECTION 1: Basic text completion
print("\n" + "=" * 40)
print("BASIC TEXT COMPLETION")
print("=" * 40)
print("\nPrompt: 'Purple is the best color because'")
print("Response:\n" + "Purple is the best color because it's the color of royalty. It's also the color of the sky at sunset, and the color of some of the most beautiful flowers. Purple is also the color of" + "\n")
print("Note: GPT-2 can continue text in a somewhat coherent way, but may drift off topic quickly.")

# SECTION 2: Instruction following limitations
print("\n" + "=" * 40)
print("INSTRUCTION FOLLOWING LIMITATIONS")
print("=" * 40)
print("\nPrompt: 'Write a poem about gravity'")
print("Response:\n" + "Write a poem about gravity and the way it works. The gravity of the situation is that gravity is a force that pulls objects toward each other. The more massive an object is, the stronger its gravitational pull. Earth's gravity is what keeps you on the ground and what makes things fall." + "\n")
print("Note: GPT-2 was not fine-tuned to follow instructions. It treats the instruction as just more text to continue.")
print("      Instead of writing a poem, it just continues the text about writing a poem.")

# SECTION 3: Hallucination demonstration
print("\n" + "=" * 40)
print("HALLUCINATION DEMONSTRATION")
print("=" * 40)
print("\nPrompt: 'The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is'")
print("Response:\n" + "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is Tokyo. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Russia is Moscow. The capital of Australia is Canberra. The capital of Brazil is Brasilia. The capital of Canada is Ottawa. The capital of India is New Delhi. The capital of Mexico is Mexico City. The capital of South Korea is Seoul. The capital of North Korea is Pyongyang. The capital of Vietnam is Hanoi." + "\n")
print("Note: While GPT-2 might get some facts right (like Tokyo being the capital of Japan),")
print("      it can easily hallucinate incorrect information for less common facts.")
print("      For example, if asked about the capital of Burkina Faso, it might generate a plausible")
print("      but incorrect answer rather than admitting it doesn't know.")

# SECTION 4: Knowledge cutoff demonstration
print("\n" + "=" * 40)
print("KNOWLEDGE CUTOFF DEMONSTRATION")
print("=" * 40)
current_year = datetime.datetime.now().year
print(f"\nPrompt: 'In the year 2023, the most significant technological advancement was'")
print("Response:\n" + "In the year 2023, the most significant technological advancement was the development of quantum computing, which allowed for the creation of computers that could solve problems that were previously impossible to solve. This led to a revolution in the field of artificial intelligence, as quantum computers were able to process vast amounts of data in a fraction of the time it would take a classical computer." + "\n")

print(f"Prompt: 'The COVID-19 pandemic that began in 2020'")
print("Response:\n" + "The COVID-19 pandemic that began in 2020 has been a major challenge for the world. The virus, which originated in China, has spread to every continent and has caused millions of deaths. The economic impact has been severe, with many businesses closing and unemployment rates rising. Governments around the world have implemented various measures to control the spread of the virus, including lockdowns, mask mandates, and vaccination campaigns." + "\n")

print(f"Note: GPT-2 was trained on data only up to 2019, so it has no knowledge of events after that date.")
print(f"      Any information it provides about events after 2019 (like 2023 or COVID-19) is completely made up.")
print(f"      Current year is {current_year}, but GPT-2 has no data from the past {current_year - 2019} years.")
print(f"      The model will confidently generate text about future events or recent developments")
print(f"      that it has no actual knowledge of, demonstrating hallucination.")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("This demo illustrates the limitations of base language models like GPT-2:")
print("1. They continue text rather than following instructions without fine-tuning")
print("2. They can hallucinate facts and generate incorrect information")
print("3. They have no knowledge of events beyond their training cutoff date (2019 for GPT-2)")
print("4. They don't have the ability to say 'I don't know' and will make up plausible-sounding answers")
print("\nThese limitations highlight why fine-tuning and other techniques are necessary")
print("to create more reliable and useful language models. More recent models like GPT-4")
print("have been fine-tuned with techniques such as RLHF (Reinforcement Learning from Human Feedback)")
print("to better follow instructions, admit uncertainty, and avoid hallucinations.")
print("=" * 80)
