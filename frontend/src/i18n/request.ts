import { getRequestConfig } from 'next-intl/server';
import esMessages from '../../messages/es.json';
import enMessages from '../../messages/en.json';
import ptMessages from '../../messages/pt.json';

const messagesMap: Record<string, any> = {
  es: esMessages,
  en: enMessages,
  pt: ptMessages
};

export default getRequestConfig(async ({ requestLocale }) => {
  let locale = await requestLocale;

  if (!locale || !['es', 'en', 'pt'].includes(locale)) {
    locale = 'es';
  }

  return {
    locale,
    messages: messagesMap[locale] || esMessages
  };
});
