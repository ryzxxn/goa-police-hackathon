import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, FileVideo, Image, Link as LinkIcon, FileText, Upload } from "lucide-react"
import Link from "next/link"

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-gray-950 text-gray-100">
      <header className="px-4 lg:px-6 h-14 flex items-center border-b border-gray-800">
        <Link className="flex items-center justify-center" href="/">
          <AlertTriangle className="h-6 w-6 text-red-500" />
          <span className="ml-2 text-lg font-bold">RadicalDetect</span>
        </Link>
        <nav className="ml-auto flex gap-4 sm:gap-6">
          <Link href="/upload">
            <Button className="bg-red-600 hover:bg-red-700 text-white">
              <Upload className="mr-2 h-4 w-4" />
              Upload Content
            </Button>
          </Link>
        </nav>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48 bg-gray-900">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="space-y-2">
                <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                  Detect Radical Content in Seconds
                </h1>
                <p className="mx-auto max-w-[700px] text-gray-400 md:text-xl">
                  Our advanced AI analyzes videos, images, YouTube links, and text documents to identify potentially radical content.
                </p>
              </div>
              <div className="space-x-4">
                <Link href="/upload">
                  <Button className="bg-red-600 hover:bg-red-700 text-white">Get Started</Button>
                </Link>
                <Link href="#features">
                  <Button variant="outline" className="text-gray-300 border-gray-600 hover:bg-gray-800">Learn More</Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
        <section id="features" className="w-full py-12 md:py-24 lg:py-32 bg-gray-950">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">Supported Content Types</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center text-gray-100">
                    <FileVideo className="mr-2 h-6 w-6 text-red-500" />
                    Video Files
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-400">Upload video files for frame-by-frame analysis and audio transcription.</p>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center text-gray-100">
                    <Image className="mr-2 h-6 w-6 text-red-500" />
                    Images
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-400">Analyze images for visual cues and embedded text related to radical content.</p>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center text-gray-100">
                    <LinkIcon className="mr-2 h-6 w-6 text-red-500" />
                    YouTube Links
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-400">Provide YouTube video links for remote content analysis without downloading.</p>
                </CardContent>
              </Card>
              <Card className="bg-gray-900 border-gray-800">
                <CardHeader>
                  <CardTitle className="flex items-center text-gray-100">
                    <FileText className="mr-2 h-6 w-6 text-red-500" />
                    Text Documents
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-400">Upload text documents for linguistic analysis and content evaluation.</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-900">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
                  Start Detecting Today
                </h2>
                <p className="mx-auto max-w-[600px] text-gray-400 md:text-xl">
                  Protect your platform and users from radical content with our advanced detection system.
                </p>
              </div>
              <div className="w-full max-w-sm space-y-2">
                <Link href="/upload">
                  <Button className="w-full bg-red-600 hover:bg-red-700 text-white">Upload Content Now</Button>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
      <footer className="flex flex-col gap-2 sm:flex-row py-6 w-full shrink-0 items-center px-4 md:px-6 border-t border-gray-800">
        <p className="text-xs text-gray-400">
          © 2024 RadicalDetect. All rights reserved.
        </p>
        <nav className="sm:ml-auto flex gap-4 sm:gap-6">
          <Link className="text-xs hover:underline underline-offset-4 text-gray-400" href="#">
            Terms of Service
          </Link>
          <Link className="text-xs hover:underline underline-offset-4 text-gray-400" href="#">
            Privacy
          </Link>
        </nav>
      </footer>
    </div>
  )
}