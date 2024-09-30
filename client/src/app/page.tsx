'use client'
import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Upload, Video, Link as LinkIcon, CheckCircle } from 'lucide-react'

export default function ContentDetectionPage() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [youtubeLink, setYoutubeLink] = useState('')
  const [textInput, setTextInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleFileDrop = (acceptedFiles: File[], setFile: React.Dispatch<React.SetStateAction<File | null>>) => {
    setFile(acceptedFiles[0] || null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      if (youtubeLink) {
        const response = await axios.post('http://127.0.0.1:5000/download', { video_url: youtubeLink })
        if (response.status === 200) {
          window.location.href = `/analysis/${response.data.vid}`
        }
      } else if (imageFile) {
        const formData = new FormData()
        formData.append('file', imageFile)

        const response = await axios.post('http://127.0.0.1:5000/image', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        if (response.status === 200) {
          window.location.href = `/analysis/${response.data.vid}`
        }
      } else if (videoFile) {
        const formData = new FormData()
        formData.append('file', videoFile)

        const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        if (response.status === 200) {
          window.location.href = `/analysis/${response.data.vid}`
        }
      } else if (textInput) {
        const response = await axios.post('http://127.0.0.1:5000/text', { text: textInput })
        if (response.status === 200) {
          window.location.href = `/analysis/${response.data.vid}`
        }
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  const isButtonDisabled = !imageFile && !videoFile && !youtubeLink && !textInput

  const { getRootProps: getImageRootProps, getInputProps: getImageInputProps } = useDropzone({
    accept: { 'image/*': [] },
    onDrop: (acceptedFiles) => handleFileDrop(acceptedFiles, setImageFile),
  })

  const { getRootProps: getVideoRootProps, getInputProps: getVideoInputProps } = useDropzone({
    accept: { 'video/*': [] },
    onDrop: (acceptedFiles) => handleFileDrop(acceptedFiles, setVideoFile),
  })

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-red-600 text-gray-100 relative gap-7">
      <div className="h-screen flex justify-center items-center">
        <div className='flex items-center fadeIN'>
          <img
            src="https://i.imgur.com/UshKh1W.png"
            alt="Background"
            className="min-h-[10rem] inset-0 object-contain w-full h-[10rem] opacity-60"
          />
          <h1 className="text-4xl font-bold text-center text-red-500 z-10 italic">RADWATCH</h1>
        </div>
      </div>

      <div className="flex flex-wrap w-full items-center justify-center">
          <div className=" p-4 animate-fadeIn">
            <CheckCircle className="text-red-500 w-8 h-8 mx-auto" />
            <h3 className="text-lg font-bold text-gray-100 text-center">Faster</h3>
            <p className="text-gray-400 text-center">Utilizing edge technologies for quick processing.</p>
          </div>
          <div className=" p-4 max-w-xs  animate-fadeIn">
            <CheckCircle className="text-red-500 w-8 h-8 mx-auto" />
            <h3 className="text-lg font-bold text-gray-100 text-center">Better Reasoning</h3>
            <p className="text-gray-400 text-center">Advanced algorithms for more accurate analysis.</p>
          </div>
          <div className=" p-4 max-w-xs  animate-fadeIn">
            <CheckCircle className="text-red-500 w-8 h-8 mx-auto" />
            <h3 className="text-lg font-bold text-gray-100 text-center">Fully Local</h3>
            <p className="text-gray-400 text-center">Complete processing on your device for privacy.</p>
          </div>
          <div className=" p-4 max-w-xs  animate-fadeIn">
            <CheckCircle className="text-red-500 w-8 h-8 mx-auto" />
            <h3 className="text-lg font-bold text-gray-100 text-center">Universal Pipeline</h3>
            <p className="text-gray-400 text-center">A highly adaptable reasoning and processing framework.</p>
          </div>
        </div>

      <div className="container mx-auto py-8 px-4">
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg p-8 max-w-2xl mx-auto border-gray-700">
          <h2 className="text-2xl font-bold text-gray-100 text-center">Content Detection Tool</h2>
          <p className="text-gray-400 text-center">
            Upload files, images, or provide a YouTube link to detect radical and hateful content.
          </p>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-gray-200 flex items-center">
                <Upload className="mr-2" />
                Upload Image
              </label>
              <div {...getImageRootProps()} className=" p-4 text-center  rounded cursor-pointer backdrop-blur-lg bg-white bg-opacity-10">
                <input {...getImageInputProps()} />
                <p className="text-gray-200">Drag & drop an image here, or click to select one</p>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-gray-200 flex items-center">
                <Video className="mr-2" />
                Upload Video File
              </label>
              <div {...getVideoRootProps()} className=" p-4 text-center  rounded cursor-pointer backdrop-blur-lg bg-white bg-opacity-10">
                <input {...getVideoInputProps()} />
                <p className="text-gray-200">Drag & drop a video here, or click to select one</p>
              </div>
            </div>
            <div className="space-y-2">
              <label htmlFor="youtube-link" className="text-gray-200 flex items-center">
                <LinkIcon className="mr-2" />
                YouTube Link
              </label>
              <input
                id="youtube-link"
                type="url"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeLink}
                onChange={(e) => setYoutubeLink(e.target.value)}
                className=" text-gray-200  focus:ring-red-500 focus:border-red-500 p-2 rounded backdrop-blur-lg bg-white bg-opacity-10"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="text-input" className="text-gray-200">Enter Text</label>
              <textarea
                id="text-input"
                placeholder='Enter text here...'
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                className=" text-gray-200  focus:ring-red-500 focus:border-red-500 p-2 rounded w-full h-32 backdrop-blur-lg bg-white bg-opacity-10"
              />
            </div>
            <button type="submit" disabled={isButtonDisabled || loading} className="w-full bg-gray-50 hover:bg-red-700 text-black font-bold p-2 rounded">
              {loading ? 'Loading...' : 'Analyze Content'}
            </button>
          </form>
        </div>
        <div className="mt-4 text-center text-sm text-gray-400 flex items-center justify-center">
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
            <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2" />
          </svg>
          This tool is for educational purposes only. Always respect privacy and legal guidelines.
        </div>
      </div>
    </div>
  )
}