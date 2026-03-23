// Test harness for imagineAI v1.2 - Run in browser console

const TEST_SUITE = [
    // SPEED questions
    { q: "How fast do birds fly?", expect: ["mph", "km/h", "speed", "fastest"] },
    { q: "How fast is the speed of light?", expect: ["299", "km/s", "meters per second"] },
    { q: "How fast do cheetahs run?", expect: ["mph", "km/h", "70", "120", "fastest"] },
    
    // COUNT questions
    { q: "How many legs does a spider have?", expect: ["eight", "8"] },
    { q: "How many legs does a dog have?", expect: ["four", "4"] },
    { q: "How many planets are in our solar system?", expect: ["eight", "8"] },
    
    // DEFINITION questions
    { q: "What is quantum mechanics?", expect: ["physics", "particles", "wave", "quantum"] },
    { q: "What is photosynthesis?", expect: ["light", "energy", "plants", "chlorophyll"] },
    { q: "What is the speed of light?", expect: ["299", "km", "meters"] },
    
    // WHO questions
    { q: "Who invented the telephone?", expect: ["Bell", "Alexander", "1876"] },
    { q: "Who discovered penicillin?", expect: ["Fleming", "1928"] },
    
    // WHY questions
    { q: "Why is the sky blue?", expect: ["scatter", "Rayleigh", "wavelength", "atmosphere"] },
    { q: "Why do leaves change color?", expect: ["chlorophyll", "pigment", "fall", "autumn"] },
    
    // WHERE questions
    { q: "Where is the Eiffel Tower?", expect: ["Paris", "France"] },
    { q: "Where is Mount Everest?", expect: ["Nepal", "Himalaya", "Tibet"] },
    
    // PROPERTY questions
    { q: "What is the capital of France?", expect: ["Paris"] },
    { q: "What is the population of China?", expect: ["billion", "1.4"] },
    
    // SUPERLATIVE questions
    { q: "What is the largest planet?", expect: ["Jupiter"] },
    { q: "What is the fastest animal?", expect: ["cheetah", "falcon", "peregrine"] },
];

console.log("TEST SUITE:", TEST_SUITE.length, "questions");
