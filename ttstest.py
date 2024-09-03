import pyttsx3

engine = pyttsx3.init(driverName='espeak')

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[39])
engine.setProperty('rate', 150)

# engine.say("One person, near, in front of you")
engine.say("Satu orang, dekat, di depan")
engine.runAndWait()
input()

# engine.say("Two cars, far, on your right")
engine.say("Dua mobil, jauh, di kanan")
engine.runAndWait()
input()

# engine.say("Three chairs, near, on your left")
engine.say("Tiga kursi, dekat, di kiri")
engine.runAndWait()
input()

# engine.say("Four tables, 114 meters, in front of you")
engine.say("Empat meja, 114 meter, di depan")
engine.runAndWait()
input()

# engine.say("Five bottles, 140 meters, on your left")
engine.say("Lima botol, 140 meter, di kiri")
engine.runAndWait()
input()

# engine.say("Six laptops, 278 meters, on your right")
engine.say("Enam laptop, 278 meter, di kanan")
engine.runAndWait()
input()

# engine.say("One laptop and two persons, far, in front of you")
engine.say("Satu laptop, dan dua orang, jauh, di depan")
engine.runAndWait()
input()

# engine.say("Three bottles, three cars, two tables, near, in front of you")
engine.say("Tiga botol, tiga mobil, dua meja, dekat, di depan")
engine.runAndWait()
input()
