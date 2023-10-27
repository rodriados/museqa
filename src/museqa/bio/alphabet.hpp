/**
 * Museqa: Multiple Sequence Aligner using hybrid parallel computing.
 * @file The representation and encoding of biological sequences' alphabet.
 * @author Rodrigo Siqueira <rodriados@gmail.com>
 * @copyright 2023-present Rodrigo Siqueira
 */
#pragma once

#include <cctype>
#include <cstdint>

#include <museqa/environment.h>

MUSEQA_BEGIN_NAMESPACE

namespace bio::alphabet
{
    /**
     * The type of a symbol within a biological sequence alphabet.
     * @since 1.0
     */
    using symbol_t = uint8_t;

    /**
     * Indicates the minimum number of bits required by the alphabet to express
     * all of the symbols that it may represent.
     * @since 1.0
     */
    inline constexpr size_t symbol_bits = 5;

    /**
     * The enumeration of biological sequences' alphabet symbols. This enumeration
     * maps all possible symbols to their internal representation values.
     * @since 1.0
     */
    enum : symbol_t
    {
        A = 0x01,   H = 0x09,   O = 0x10,   U = 0x15
      , B = 0x05,   I = 0x0A,   P = 0x11,   V = 0x16
      , C = 0x02,   J = 0x0B,   Q = 0x12,   W = 0x17
      , D = 0x06,   K = 0x0C,   R = 0x13,   X = 0x18
      , E = 0x07,   L = 0x0D,   S = 0x14,   Y = 0x19
      , F = 0x08,   M = 0x0E,   T = 0x03,   Z = 0x1A
      , G = 0x04,   N = 0x0F

      , end     = 0x00
      , gap     = 0x1C
      , extend  = 0x1D
      , unknown = 0x1B
    };

    /**
     * Encodes a single biological symbol from character to its internal representation.
     * @param symbol The biological symbol to be encoded.
     * @return The symbol's encoded representation.
     */
    inline constexpr symbol_t encode(const char symbol) noexcept
    {
        switch (static_cast<char>(toupper(symbol))) {
            case 'A': return alphabet::A;   case 'N': return alphabet::N;
            case 'B': return alphabet::B;   case 'O': return alphabet::O;
            case 'C': return alphabet::C;   case 'P': return alphabet::P;
            case 'D': return alphabet::D;   case 'Q': return alphabet::Q;
            case 'E': return alphabet::E;   case 'R': return alphabet::R;
            case 'F': return alphabet::F;   case 'S': return alphabet::S;
            case 'G': return alphabet::G;   case 'T': return alphabet::T;
            case 'H': return alphabet::H;   case 'U': return alphabet::U;
            case 'I': return alphabet::I;   case 'V': return alphabet::V;
            case 'J': return alphabet::J;   case 'W': return alphabet::W;
            case 'K': return alphabet::K;   case 'X': return alphabet::X;
            case 'L': return alphabet::L;   case 'Y': return alphabet::Y;
            case 'M': return alphabet::M;   case 'Z': return alphabet::Z;

            case '-': return alphabet::gap;
            case '*': return alphabet::end;
            default:  return alphabet::unknown;
        }
    }

    /**
     * Decodes a single biological symbol to its corresponding character.
     * @param symbol The biological symbol to be decoded into corresponding character.
     * @return The symbol's character representation.
     */
    inline constexpr char decode(const symbol_t symbol) noexcept
    {
        switch (symbol) {
            case alphabet::A: return 'A';   case alphabet::N: return 'N';
            case alphabet::B: return 'B';   case alphabet::O: return 'O';
            case alphabet::C: return 'C';   case alphabet::P: return 'P';
            case alphabet::D: return 'D';   case alphabet::Q: return 'Q';
            case alphabet::E: return 'E';   case alphabet::R: return 'R';
            case alphabet::F: return 'F';   case alphabet::S: return 'S';
            case alphabet::G: return 'G';   case alphabet::T: return 'T';
            case alphabet::H: return 'H';   case alphabet::U: return 'U';
            case alphabet::I: return 'I';   case alphabet::V: return 'V';
            case alphabet::J: return 'J';   case alphabet::W: return 'W';
            case alphabet::K: return 'K';   case alphabet::X: return 'X';
            case alphabet::L: return 'L';   case alphabet::Y: return 'Y';
            case alphabet::M: return 'M';   case alphabet::Z: return 'Z';

            case alphabet::gap:     [[fallthrough]];
            case alphabet::extend:  return '-';
            case alphabet::end:     return '*';
            default:                return '#';
        }
    }
}

MUSEQA_END_NAMESPACE