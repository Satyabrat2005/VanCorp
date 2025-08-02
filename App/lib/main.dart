import 'package:flutter/material.dart';
import 'dart:math';

void main() => runApp(const VancorpDeluxeApp());

class VancorpDeluxeApp extends StatelessWidget {
  const VancorpDeluxeApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VANITY by Vancorp',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  bool quoteVisible = true;
  bool carouselVisible = false;
  bool brandVisible = false;

  late final PageController pageController;
  late final AnimationController quoteController;
  late final Animation<double> quoteOpacity;

  @override
  void initState() {
    super.initState();
    pageController = PageController(viewportFraction: 0.7);

    quoteController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    );
    quoteOpacity = Tween<double>(begin: 0, end: 1).animate(
        CurvedAnimation(parent: quoteController, curve: Curves.easeInOut));

    quoteController.forward();

    Future.delayed(const Duration(seconds: 4), () {
      quoteController.reverse();
      setState(() => quoteVisible = false);
    });

    Future.delayed(const Duration(seconds: 5), () {
      setState(() => carouselVisible = true);
    });

    Future.delayed(const Duration(seconds: 9), () {
      setState(() => brandVisible = true);
    });
  }

  @override
  void dispose() {
    pageController.dispose();
    quoteController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0C0C0E),
      body: Stack(children: [
        if (quoteVisible)
          Center(
            child: FadeTransition(
              opacity: quoteOpacity,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32.0),
                child: Text(
                  '"Empowering the shape of ecommerce with Vancorp Holdings"',
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 24,
                    fontWeight: FontWeight.w300,
                    height: 1.5,
                    letterSpacing: 1.2,
                    fontStyle: FontStyle.italic,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),

        if (carouselVisible)
          Positioned.fill(
            child: PageView.builder(
              controller: pageController,
              itemCount: sampleImages.length,
              itemBuilder: (context, index) {
                return AnimatedBuilder(
                  animation: pageController,
                  builder: (context, child) {
                    double value = 0;
                    if (pageController.position.hasContentDimensions) {
                      value = pageController.page! - index;
                    }
                    value = (1 - value.abs() * 0.3).clamp(0.0, 1.0);
                    double rotateY = (pageController.page! - index) * pi / 8;
                    return Transform(
                      transform: Matrix4.identity()
                        ..setEntry(3, 2, 0.001)
                        ..rotateY(rotateY)
                        ..scale(value, value),
                      alignment: Alignment.center,
                      child: child,
                    );
                  },
                  child: Container(
                    margin: const EdgeInsets.symmetric(
                        vertical: 100, horizontal: 12),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade900,
                      borderRadius: BorderRadius.circular(24),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.7),
                          blurRadius: 16,
                          offset: const Offset(0, 8),
                        ),
                      ],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: Image.asset(
                        sampleImages[index],
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

        if (brandVisible)
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 80),
              child: Text(
                'VANCORP',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 42,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 8,
                  shadows: [
                    Shadow(
                      color: Colors.white24,
                      blurRadius: 12,
                    ),
                  ],
                ),
              ),
            ),
          ),

        Positioned(
          top: 40,
          left: 40,
          child: Text(
            'VANITY',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.bold,
              letterSpacing: 2,
            ),
          ),
        ),
      ]),
    );
  }
}

// Sample assets list
const sampleImages = [
  'assets/cloth1.jpg',
  'assets/cloth2.jpg',
  'assets/cloth3.jpg',
];
