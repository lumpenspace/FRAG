# NextJS Fren

## Summarizer

`SYS`
> The summarizing bot is an intelligent filter and condenser, adept in recognizing pertinent information from Next.js and React content. It is given a user question and retrieved document. It first determines whether the retrieved text is relevant to the user's question. If relevant, the bot then produces a clear, succinct summary designed to assist the interface bot in answering the user query effectively. The bot's output is structured into two parts: the first confirms the relevance of the content (# Relevance), and the second provides a concise summary (# Summary). The summary should provide key points and essential context that allow the interface bot to generate informative and precise responses.```

### Example: relevant

`USR`
> cannot seem to link assets in pages with dynamic paths

`USR`

> ```yml
> type: document
> url: https://nextjs.org/docs/pages/building-your-application/optimizing/optimizing-images"
>```
>
> getStaticPath
  If a page has Dynamic Routes and uses getStaticProps, it needs to define a list of paths to be statically generated.
> When you export a function called getStaticPaths (Static Site Generation) from a page that uses dynamic routes, Next.js will statically pre-render all the paths specified by getStaticPaths.
>pages/repo/[name].tsx
>
>  ```ts
>  import type {
>    InferGetStaticPropsType,
>    GetStaticProps,
>    GetStaticPaths,
>  } from 'next'
>  type Repo = {
>    name: string
>    stargazers_count: number
>  }
>  export const getStaticPaths = (async () => {
>  return {
>    paths: [
>      {
>        params: {
>          name: 'next.js',
>        },
>      }, // See the "paths" section below
>    ],
>    fallback: true, // false or "blocking"
>  }
>  }) satisfies GetStaticPaths
>  export const getStaticProps = (async (context) => {
>    const res = await fetch('https://api.github.com/repos/vercel/next.js')
>    const repo = await res.json()
>    return { props: { repo } }
>  }) satisfies GetStaticProps<{
>    repo: Repo
>  }>
>  export default function Page({
>    repo,
>  }: InferGetStaticPropsType<typeof getStaticProps>) {
>    return repo.stargazers_count
>  }
> ```
>
> The getStaticPaths API reference covers all parameters and props that can be used with getStaticPaths.
> When should I use getStaticPaths?
> You should use getStaticPaths if you’re statically pre-rendering pages that use dynamic routes and:
> The data comes from a headless CMS
> The data comes from a database
> The data comes from the filesystem
> The data can be publicly cached (not user-specific)
> The page must be pre-rendered (for SEO) and be very fast — getStaticProps generates HTML and JSON files, both of which can be cached by a CDN for performance
> When does getStaticPaths run
> getStaticPaths will only run during build in production, it will not be called during runtime. You can validate code written inside getStaticPaths is removed from the client-side bundle with this tool.
> How does getStaticProps run with regards to getStaticPaths
> getStaticProps runs during next build for any paths returned during build
> getStaticProps runs in the background when using fallback: true
> getStaticProps is called before initial render when using fallback: blocking
> Where can I use getStaticPaths
> getStaticPaths must be used with getStaticProps
> You cannot use getStaticPaths with getServerSideProps
> You can export getStaticPaths from a Dynamic Route that also uses getStaticProps
> You cannot export getStaticPaths from non-page file (e.g. your components folder)
> You must export getStaticPaths as a standalone function, and not a property of the page component
> Runs on every request in development
> In development (next dev), getStaticPaths will be called on every request.
> Generating paths on-demand
> getStaticPaths allows you to control which pages are generated during the build instead of on-demand with fallback. Generating more pages during a build will cause slower builds.
> You can defer generating all pages on-demand by returning an empty array for paths. This can be especially helpful when deploying your Next.js application to multiple environments. For example, you can have faster builds by generating all pages on-demand for previews (but not production builds). This is helpful for sites with hundreds/thousands of static pages.
>
>  ```ts
>  pages/posts/[id].js
>  export async function getStaticPaths() {
>    // When this is true (in preview environments) don't
>    // prerender any static pages
>    // (faster builds, but slower initial page load)
>    if (process.env.SKIP_BUILD_STATIC_GENERATION) {
>      return {
>        paths: [],
>        fallback: 'blocking',
>      }
>    }
>    // Call an external API endpoint to get posts
>    const res = await fetch('https://.../posts')
>    const posts = await res.json()
>    // Get the paths we want to prerender based on posts
>    // In production environments, prerender all pages
>    // (slower builds, but faster initial page load)
>    const paths = posts.map((post) => ({
>      params: { id: post.id },
>    }))
>    // { fallback: false } means other routes should 404
>    return { paths, fallback: false }
>  }

`BOT`
>
> ```yml
> type: summary
> relevance: low
> ```
>
> The discussion on getStaticPaths and getStaticProps in Next.js is focused on static generation and dynamic routing. These functions are key for optimizing SEO and performance through the pre-rendering of pages that use dynamic routes. Although they are crucial for understanding static site generation, they do not address the specific challenge of linking assets in pages with dynamic paths. The user's issue might rather be related to project structure and how assets are managed within the Next.js environment.

## Example: Irrelevant

`USR`
> i cannot seem to link assets in pages with dynamic paths

`USR`
>
> ```yml
> type: document,
> url: "https://nextjs.org/docs/pages/building-your-application/optimizing/optimizing-images" 
>```
>
> Next.js can serve static files, like images, under a folder called public in the root directory. Files inside public can then be referenced by your code starting from the base URL (/).
> For example, the file public/avatars/me.png can be viewed by visiting the /avatars/me.png path. The code to display that image might look like:
>
> ```tsx
>
> import Image from 'next/image'
> export function Avatar({ id, alt }) {
>   return <Image src={`/avatars/${id}.png`} alt={alt} width="64" height="64" />  
> }
> export function AvatarOfMe() {
>   return <Avatar id="me" alt="A portrait of me" />
> }
>
> ```
>
> Caching
> Next.js cannot safely cache assets in the public folder because they may change. The default caching headers applied are:
> Cache-Control: public, max-age=0
> Robots, Favicons, and others
> The folder is also useful for robots.txt, favicon.ico, Google Site Verification, and any other static files (including .html). But make sure to not have a static file with the same name as a file in the pages/ directory, as this will result in an error. Read more.
> Good to know:
> The directory must be named public. The name cannot be changed and it's the only directory used to serve static assets.
> Only assets that are in the public directory at build time will be served by Next.js. Files added at request time won't be available. We recommend using a third-party service like Vercel Blob for persistent file storage.

`BOT`
>
> ```yml
> type: summar
> relevance: hig
> }
> ```
>
> Next.js allows for serving static files, such as images, from a folder named public located in the root directory. Files in this folder can be accessed directly using a URL path relative to the base, like /avatars/me.png. To display an image from the public folder, you might use the Image component from Next.js, as demonstrated in the Avatar component example where the image source is dynamically constructed based on the id of the avatar. This method ensures that assets are efficiently managed and referenced within Next.js projects. Note that static files like robots.txt and favicon.ico should also be placed in this directory to avoid naming conflicts with the pages directory. Additionally, Next.js sets default caching headers for these assets to Cache-Control: public, max-age=0 because they might change and cannot be cached safely by Next.js itself.

## Interface

`SYS`
>
> The assistant is a friendly programmer, expert in Next.JS and React. He will receive user questions along with a context, the latter marked as #knowledge, and it will use them to answer the user query. The assistant response will be structured in two parts, the first part is the answer to the user query, and the second part is the reasoning behind the answer. They will be marked as `# Answer` and `# Reasoning`

`USR`
> Which one should I use, /app or /pages?

`USR`

> ```yml
> type: summary
> title: App Router Migration
> source: https://nextjs.org/docs/pages/building-your-application/upgrading/app-router-migration
> relevance: lo
> ```
>
> Migration Strategy:
>
> Coexistence: /app and /pages can be used simultaneously.
> Incremental Migration: Facilitates page-by-page migration, reducing complexity.
> Directory Features:
> /app:
> Nested Routes: Supports complex nested routing structures.
> Layouts: Uses layout.js for shared UI across routes.
> Special Files: Uses page.js for route-specific UI.
> File Types: Allows .js, .jsx, .tsx.
> Additional Files: Can include components, styles, tests, etc.
> Data Fetching: New API replaces getServerSideProps and getStaticProps.
> Error Handling: Uses error.js and not-found.js for errors.
> Root Layout: Replaces pages/_app.js and pages/_document.js with app/layout.js.
> SEO: Built-in support, replacing next/head.
> /pages:
> Traditional Routing: Basic file-based routing.
> Standard Error Pages: Uses _error.js, 404.js.
> API Routes: Handled by files in pages/api/.
> Head Management: Uses next/head for HTML `<head>` elements.
> Migration Steps:
> Update Next.js: Ensure version 13.4 or greater.
> Create /app Directory: At project root or inside /src.
> Set Up Root Layout: Create app/layout.tsx for universal layout configuration.

`USR`

> ```yml
> type: summary
> title: App Routerr Migration
> source: https://nextjs.org/docs/pages/building-your-application/upgrading/app-router-migration
> relevance: low
> ```
>
> Historical Context:
> Previous Structure: Typically, pages were stored in the pages directory.
> Example: └── pages ├── about.js ├── index.js └── team.js
> New App Router Introduction:
>
>? Current Structure: With the App Router, the structure shifts to a more modular approach under src/app.
> Example: src/ └── app ├── about │ └── page.js ├── globals.css ├── layout.js ├── login │ └── page.js ├── page.js └── team └── route.js
> Directory and Routing Conventions:
>
> Page Directory as URL Path: Each page in the App Router has its own directory, dictating the URL path.
> Rendered Component: page.js in a directory is the main component rendered for that path.
> Additional Components: Other components can be stored in the directory but won't affect routing unless named page.js.
> Reserved Files: Specific files like loading.js, template.js, and layout.js have designated functionalities.
> Key Features of App Router:
>
> Component Types:
> Server Components: By default, components in the app directory are treated as server components, enhancing server-side rendering capabilities.
> Exclusive Features: Certain features, newly introduced or enhanced, are only available in the App Router and not in the traditional Pages Router.
> This summary structures the key elements from the article into a format suitable for bots to process and retrieve information efficiently for users or further bot-to-bot interactions. This approach ensures clarity and direct access to structured data, which can be beneficial in automated systems and databases.
>
