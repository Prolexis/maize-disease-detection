import { NextResponse } from 'next/server';
import createMiddleware from 'next-intl/middleware';

const intlMiddleware = createMiddleware({
  // Locales soportados
  locales: ['es', 'en', 'pt'],
  // Locale por defecto — redirige / -> /es
  defaultLocale: 'es',
  // Siempre prefijar con el locale en la URL
  localePrefix: 'always'
});

export function proxy(request: any) {
  try {
    return intlMiddleware(request);
  } catch (error) {
    console.error("--> PROXY ERROR:", error);
    return NextResponse.next();
  }
}

export default proxy;

export const config = {
  // Aplicar el proxy en todas las rutas excepto las estáticas y de API
  matcher: ['/((?!api|_next|_vercel|.*\\..*).*)', '/']
};
