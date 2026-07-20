import { getRequestConfig } from 'next-intl/server';
import esMessages from '../../messages/es.json';
import enMessages from '../../messages/en.json';
import ptMessages from '../../messages/pt.json';

const messagesMap: Record<string, any> = {
  es: esMessages,
  en: enMessages,
  pt: ptMessages
};

export default getRequestConfig(async (params) => {
  const reqLocale = await (params as any).requestLocale;
  const locale = (reqLocale && ['es', 'en', 'pt'].includes(reqLocale)) ? reqLocale : 'es';

  return {
    locale,
    messages: messagesMap[locale] || esMessages
  };
});
