(function () {

	// Enable transitions only after first paint (prevents load-time animation flash).
	window.addEventListener('load', function () {
		setTimeout(function () {
			document.body.classList.remove('is-preload');
		}, 100);
	});

	// Email de-obfuscation: the address is stored reversed in the HTML.
	document.querySelectorAll('.email-obf').forEach(function (span) {
		var email = span.textContent.trim().split('').reverse().join('');
		var a = document.createElement('a');
		a.href = 'mailto:' + email;
		a.textContent = email;
		span.replaceWith(a);
	});

	// Nav scroll-spy: the section crossing the viewport's vertical center is active.
	var links = document.querySelectorAll('#nav a[href^="#"]');
	var observer = new IntersectionObserver(function (entries) {
		entries.forEach(function (entry) {
			if (!entry.isIntersecting) return;
			links.forEach(function (link) {
				link.classList.toggle('active', link.hash === '#' + entry.target.id);
			});
		});
	}, { rootMargin: '-50% 0px -50% 0px' });

	links.forEach(function (link) {
		var section = document.querySelector(link.hash);
		if (section) observer.observe(section);
		link.addEventListener('click', function () {
			links.forEach(function (l) { l.classList.remove('active'); });
			link.classList.add('active');
			document.body.classList.remove('header-visible');
		});
	});

	// Mobile sidebar toggle.
	var toggle = document.querySelector('#headerToggle .toggle');
	toggle.addEventListener('click', function (e) {
		e.preventDefault();
		document.body.classList.toggle('header-visible');
	});
	document.getElementById('main').addEventListener('click', function () {
		document.body.classList.remove('header-visible');
	});

})();
