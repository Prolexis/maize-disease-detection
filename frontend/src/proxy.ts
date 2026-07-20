import createMiddleware from 'next-intl/middleware';
import { routing } from './i18n/routing';

const intlMiddleware = createMiddleware(routing);

export function proxy(request: any) {
  return intlMiddleware(request);
}

export default proxy;

export const config = {
  matcher: ['/', '/(es|en|pt)', '/(es|en|pt)/:path*']
};
