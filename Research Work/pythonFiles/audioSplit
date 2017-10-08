import wave

origAudio = wave.open('audio.wav','r')
frameRate = origAudio.getframerate()
nChannels = origAudio.getnchannels()
sampWidth = origAudio.getsampwidth()

start = float(5)
end = float(21)

origAudio.setpos(start*frameRate)
chunkData = origAudio.readframes(int((end-start)*frameRate))

chunkAudio = wave.open('outputFile1.wav','w')
chunkAudio.setnchannels(nChannels)
chunkAudio.setsampwidth(sampWidth)
chunkAudio.setframerate(frameRate)
chunkAudio.writeframes(chunkData)
chunkAudio.close()
