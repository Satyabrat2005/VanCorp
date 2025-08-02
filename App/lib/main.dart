import 'package:flutter/material.dart';

void main() {
  runApp(const VancorpApp());
}

class VancorpApp extends StatelessWidget {
  const VancorpApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Vancorp Holdings',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const LandingPage(),
    );
  }
}

class LandingPage extends StatefulWidget {
  const LandingPage({super.key});

  @override
  State<LandingPage> createState() => _LandingPageState();
}

class _LandingPageState extends State<LandingPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _offsetAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1600),
      vsync: this,
    );

    _offsetAnimation = Tween<Offset>(
      begin: const Offset(-1.5, -1.5),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
    ));

    _scaleAnimation = Tween<double>(
      begin: 0.2,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOut,
    ));

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _goToNextPage() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const LocomotivePage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0B0B0D),
      body: Stack(
        children: [
          // Glowing circle animation
          Positioned(
            top: 60,
            left: 60,
            child: SlideTransition(
              position: _offsetAnimation,
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Container(
                  width: 140,
                  height: 140,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.transparent,
                    border: Border.all(color: Colors.white70, width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.white.withOpacity(0.3),
                        blurRadius: 40,
                        spreadRadius: 8,
                      ),
                    ],
                  ),
                  child: const Center(
                    child: Text(
                      'VANITY',
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                        letterSpacing: 2,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),

          // Centered Quote
          Align(
            alignment: Alignment.center,
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28.0),
              child: Text(
                '"Empowering the shape of ecommerce\nwith Vancorp Holdings"',
                style: const TextStyle(
                  color: Color(0xFFB4B7BC),
                  fontSize: 26,
                  fontWeight: FontWeight.w300,
                  height: 1.5,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),

          // Button to go to second page
          Positioned(
            bottom: 80,
            left: 0,
            right: 0,
            child: Center(
              child: ElevatedButton(
                onPressed: _goToNextPage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(
                      horizontal: 40, vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  "Explore",
                  style: TextStyle(
                    fontSize: 16,
                    letterSpacing: 1.2,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class LocomotivePage extends StatelessWidget {
  const LocomotivePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text("Locomotive"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: const Center(
        child: Text(
          "This is your second page.\nInspired by smooth locomotive transitions.",
          style: TextStyle(
            color: Colors.white70,
            fontSize: 22,
            height: 1.6,
          ),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}
