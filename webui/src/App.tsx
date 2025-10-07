import { useState } from 'react'
import type { ChangeEvent } from 'react'

const API_KEY = '97ce09205a014871bb8ee119a921137e'
const API_URL = 'https://api.fish.audio/v1/asr'
const TTS_API_URL = 'https://api.fish.audio/v1/tts'

const languages = [
  { label: 'English', value: 'en' },
  { label: 'Mandarin', value: 'zh-CN' },
  { label: 'Cantonese', value: 'yue' }
]

const voiceTypes = [
  { label: 'Neutral', value: 'neutral' },
  { label: 'Friendly', value: 'friendly' },
  { label: 'Professional', value: 'professional' },
  { label: 'Energetic', value: 'energetic' },
  { label: 'Calm', value: 'calm' },
  { label: 'Storyteller', value: 'storyteller' }
]

function App() {
  const [audio, setAudio] = useState<File | null>(null)
  const [language, setLanguage] = useState('en')
  const [transcript, setTranscript] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  // TTS states
  const [text, setText] = useState('')
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null)
  const [referenceText, setReferenceText] = useState('')
  const [voiceType, setVoiceType] = useState('neutral')
  const [speechSpeed, setSpeechSpeed] = useState(1.0)
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null)
  const [ttsLoading, setTtsLoading] = useState(false)
  const [ttsError, setTtsError] = useState('')

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudio(e.target.files[0])
      setTranscript('')
      setError('')
    }
  }

  const handleReferenceAudioChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      // Validate audio file
      if (!file.type.startsWith('audio/')) {
        setTtsError('Please select a valid audio file')
        return
      }
      setReferenceAudio(file)
      setTtsError('')
    }
  }

  const validateAudioFile = (file: File): boolean => {
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/ogg']
    const maxSize = 10 * 1024 * 1024 // 10MB
    
    if (!validTypes.includes(file.type)) {
      setTtsError('Please select a valid audio file (WAV, MP3, FLAC, OGG)')
      return false
    }
    
    if (file.size > maxSize) {
      setTtsError('Audio file is too large. Please select a file smaller than 10MB')
      return false
    }
    
    return true
  }

  const handleTranscribe = async () => {
    if (!audio) {
      setError('Please upload an audio file.')
      return
    }
    setLoading(true)
    setError('')
    setTranscript('')
    try {
      const formData = new FormData()
      formData.append('audio', audio)
      formData.append('language', language)
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${API_KEY}`
        },
        body: formData
      })
      if (!res.ok) throw new Error('Transcription failed')
      const data = await res.json()
      setTranscript(data.transcript || 'No transcript returned.')
    } catch (err: any) {
      setError(err.message || 'An error occurred.')
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateTTS = async () => {
    if (!text.trim()) {
      setTtsError('Please enter some text to synthesize.')
      return
    }

    if (referenceAudio && !validateAudioFile(referenceAudio)) {
      return
    }

    setTtsLoading(true)
    setTtsError('')
    setGeneratedAudio(null)

    try {
      const formData = new FormData()
      formData.append('text', text)
      formData.append('voice_type', voiceType)
      formData.append('speech_speed', speechSpeed.toString())
      
      if (referenceAudio) {
        formData.append('reference_audio', referenceAudio)
      }
      
      if (referenceText.trim()) {
        formData.append('reference_text', referenceText)
      }

      const res = await fetch(TTS_API_URL, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${API_KEY}`
        },
        body: formData
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.error || 'TTS generation failed')
      }

      const audioBlob = await res.blob()
      const audioUrl = URL.createObjectURL(audioBlob)
      setGeneratedAudio(audioUrl)
    } catch (err: any) {
      setTtsError(err.message || 'An error occurred during TTS generation.')
    } finally {
      setTtsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">Fish Speech AI</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Speech-to-Text Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">🎤 Speech-to-Text</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">Upload Audio</label>
                <input 
                  type="file" 
                  accept="audio/*" 
                  onChange={handleFileChange} 
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" 
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">Language</label>
                <select
                  value={language}
                  onChange={e => setLanguage(e.target.value)}
                  className="block w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {languages.map(lang => (
                    <option key={lang.value} value={lang.value}>{lang.label}</option>
                  ))}
                </select>
              </div>
              <button
                onClick={handleTranscribe}
                disabled={loading || !audio}
                className="w-full bg-blue-600 text-white font-semibold py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Transcribing...' : 'Transcribe'}
              </button>
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">Transcript</label>
                <textarea
                  value={transcript}
                  readOnly
                  rows={4}
                  className="block w-full border border-gray-300 rounded-lg px-3 py-2 bg-gray-50 resize-none"
                  placeholder="Transcribed text will appear here..."
                />
              </div>
              {error && <div className="text-red-600 text-sm text-center bg-red-50 p-2 rounded">{error}</div>}
            </div>
          </div>

          {/* Text-to-Speech Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">🔊 Text-to-Speech</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">Text to Synthesize</label>
                <textarea
                  value={text}
                  onChange={e => setText(e.target.value)}
                  rows={3}
                  className="block w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  placeholder="Enter the text you want to convert to speech..."
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">Voice Type</label>
                <select
                  value={voiceType}
                  onChange={e => setVoiceType(e.target.value)}
                  className="block w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {voiceTypes.map(voice => (
                    <option key={voice.value} value={voice.value}>{voice.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-gray-600">
                  Speech Speed: {speechSpeed.toFixed(1)}x
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="2.0"
                  step="0.1"
                  value={speechSpeed}
                  onChange={e => setSpeechSpeed(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.5x</span>
                  <span>1.0x</span>
                  <span>2.0x</span>
                </div>
              </div>

              <div className="border-2 border-dashed border-green-300 rounded-lg p-4 bg-green-50">
                <h3 className="text-lg font-semibold text-green-800 mb-3">🎤 Voice Cloning Options</h3>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium mb-2 text-green-700">Reference Audio File</label>
                  <input 
                    type="file" 
                    accept="audio/*" 
                    onChange={handleReferenceAudioChange} 
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-green-100 file:text-green-800 hover:file:bg-green-200" 
                  />
                  <p className="text-xs text-green-600 mt-1">Upload a reference audio file for voice cloning (WAV, MP3, FLAC, OGG)</p>
                  {referenceAudio && (
                    <p className="text-xs text-green-700 mt-1 font-medium">✅ Selected: {referenceAudio.name}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2 text-green-700">Reference Text</label>
                  <textarea
                    value={referenceText}
                    onChange={e => setReferenceText(e.target.value)}
                    rows={3}
                    className="block w-full border border-green-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
                    placeholder="Enter the text that corresponds to the reference audio..."
                  />
                  <p className="text-xs text-green-600 mt-1">Enter the exact text spoken in the reference audio for better voice cloning</p>
                </div>
              </div>

              <button
                onClick={handleGenerateTTS}
                disabled={ttsLoading || !text.trim()}
                className="w-full bg-green-600 text-white font-semibold py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {ttsLoading ? 'Generating...' : 'Generate Speech'}
              </button>

              {generatedAudio && (
                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-600">Generated Audio</label>
                  <audio controls className="w-full">
                    <source src={generatedAudio} type="audio/wav" />
                    Your browser does not support the audio element.
                  </audio>
                </div>
              )}

              {ttsError && <div className="text-red-600 text-sm text-center bg-red-50 p-2 rounded">{ttsError}</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
