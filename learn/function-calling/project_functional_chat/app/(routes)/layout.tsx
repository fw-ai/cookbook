import '~/styles/globals.css';

export interface RootLayoutProps {
  readonly children: React.ReactNode;
}
export default async function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/images/favicon.ico" />
      </head>
      <body className="font-lato overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
