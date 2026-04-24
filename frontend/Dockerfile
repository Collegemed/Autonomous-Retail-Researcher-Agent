# nginx is a lightweight web server - perfect for serving HTML/CSS/JS
FROM nginx:alpine

# Copy your frontend files into nginx's serving folder
COPY index.html /usr/share/nginx/html/
COPY style.css  /usr/share/nginx/html/
COPY script.js  /usr/share/nginx/html/

# Copy our custom nginx config (we create this next)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx listens on port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]