import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin();

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configuración de i18n para next-intl con locales
};

export default withNextIntl(nextConfig);
