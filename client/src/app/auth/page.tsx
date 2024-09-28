"use client"
import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { AlertCircle } from "lucide-react"

export default function ContentDetectionPage() {
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [youtubeLink, setYoutubeLink] = useState('')
  const [textFile, setTextFile] = useState<File | null>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Handle form submission logic here
    console.log({ imageFile, videoFile, youtubeLink, textFile })
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-red-600 text-gray-100">
      <div className="h-screen flex justify-center items-center relative" >
          <div className='flex justify-between absolute top-0 w-full flex-1'>
          <div className="siren-light left-top"></div>
          <div className="siren-light right-top"></div>
          <div className="siren-light2 left-bottom"></div>
          <div className="siren-light2 right-bottom"></div>
          </div>
          <h1 className="text-4xl font-bold text-center mb-8 text-red-500 ">RadWatch</h1>
        </div>
      <div className="container mx-auto py-8 px-4">
        <Card className="max-w-2xl mx-auto bg-gray-950 border-gray-950">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-gray-100 text-center">Content Detection Tool</CardTitle>
            <CardDescription className="text-gray-400">
              Upload files, images, or provide a YouTube link to detect radical and hateful content.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="image-upload" className="text-gray-200">Upload Image</Label>
                <Input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={(e) => setImageFile(e.target.files?.[0] || null)}
                  className="bg-gray-700 text-gray-200 border-gray-600 focus:ring-red-500 focus:border-red-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="video-upload" className="text-gray-200">Upload Video File</Label>
                <Input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                  className="bg-gray-700 text-gray-200 border-gray-600 focus:ring-red-500 focus:border-red-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="youtube-link" className="text-gray-200">YouTube Link</Label>
                <Input
                  id="youtube-link"
                  type="url"
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={youtubeLink}
                  onChange={(e) => setYoutubeLink(e.target.value)}
                  className="bg-gray-700 text-gray-200 border-gray-600 focus:ring-red-500 focus:border-red-500"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="text-upload" className="text-gray-200">Upload Text Document</Label>
                <Input
                  id="text-upload"
                  type="file"
                  accept=".txt,.doc,.docx,.pdf"
                  onChange={(e) => setTextFile(e.target.files?.[0] || null)}
                  className="bg-gray-700 text-gray-200 border-gray-600 focus:ring-red-500 focus:border-red-500"
                />
              </div>
              <Button type="submit" className="w-full bg-gray-50 hover:bg-red-700 text-black font-bold">
                Analyze Content
              </Button>
            </form>
          </CardContent>
        </Card>
        <div className="mt-4 text-center text-sm text-gray-400 flex items-center justify-center">
          <AlertCircle className="w-4 h-4 mr-2" />
          This tool is for educational purposes only. Always respect privacy and legal guidelines.
        </div>
      </div>
    </div>
  )
}