import React from 'react';
import { jsPDF } from "jspdf";

interface PDFDownloadButtonProps {
  text: string;
  fileName?: string;
}

const PDFDownloadButton: React.FC<PDFDownloadButtonProps> = ({ 
  text, 
  fileName = "Summary_Report.pdf" 
}) => {
  const generatePDF = () => {
    const doc = new jsPDF();
    
    // Split the text into lines that fit within the PDF width
    const lines = doc.splitTextToSize(text, 190);
    
    // Add text to the PDF
    doc.text(lines, 10, 10);
    
    // Save and download the PDF
    doc.save(fileName);
  };

  return (
    <button 
      onClick={generatePDF}
      className="bg-gray-100 hover:bg-gray-950 text-black font-bold hover:text-white px-4 py-2 rounded transition-colors duration-200">
      Download PDF
    </button>
  );
};

export default PDFDownloadButton;