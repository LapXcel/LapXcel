import Image from "next/image"; // Import the Image component from Next.js for optimized image handling

// Define the main functional component for the Home page
export default function Home() {
  return (
    // Main container with a grid layout, responsive design, and padding
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        {/* Display the Next.js logo using the Image component */}
        <Image
          className="dark:invert" // Invert colors for dark mode
          src="/next.svg" // Source of the logo image
          alt="Next.js logo" // Alternative text for accessibility
          width={180} // Width of the image
          height={38} // Height of the image
          priority // Mark the image as high priority for loading
        />
        {/* Ordered list of instructions for getting started */}
        <ol className="list-inside list-decimal text-sm/6 text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
          <li className="mb-2 tracking-[-.01em]">
            Get started by editing{" "}
            <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-[family-name:var(--font-geist-mono)] font-semibold">
              src/app/page.tsx
            </code>
            .
          </li>
          <li className="tracking-[-.01em]">
            Save and see your changes instantly.
          </li>
        </ol>

        {/* Container for action buttons */}
        <div className="flex gap-4 items-center flex-col sm:flex-row">
          {/* Button to deploy the application on Vercel */}
          <a
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:w-auto"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank" // Open link in new tab
            rel="noopener noreferrer" // Security feature for external links
          >
            <Image
              className="dark:invert" // Invert colors for dark mode
              src="/vercel.svg" // Source of the Vercel logo
              alt="Vercel logomark" // Alternative text for accessibility
              width={20} // Width of the image
              height={20} // Height of the image
            />
            Deploy now // Button text
          </a>
          {/* Button to read the documentation */}
          <a
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent font-medium text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 w-full sm:w-auto md:w-[158px]"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read our docs // Button text
          </a>
        </div>
      </main>
      {/* Footer section with additional links */}
      <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
        {/* Link to the Next.js learning resources */}
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden // Hide the image from assistive technologies
            src="/file.svg" // Source of the file icon
            alt="File icon" // Alternative text for accessibility
            width={16} // Width of the image
            height={16} // Height of the image
          />
          Learn // Link text
        </a>
        {/* Link to Next.js templates */}
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg" // Source of the window icon
            alt="Window icon" // Alternative text for accessibility
            width={16} // Width of the image
            height={16} // Height of the image
          />
          Examples // Link text
        </a>
        {/* Link to the main Next.js website */}
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg" // Source of the globe icon
            alt="Globe icon" // Alternative text for accessibility
            width={16} // Width of the image
            height={16} // Height of the image
          />
          Go to nextjs.org → // Link text
        </a>
      </footer>
    </div>
  );
}
